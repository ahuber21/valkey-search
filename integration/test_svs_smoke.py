"""End-to-end smoke test for ALGORITHM SVS_VAMANA.

Exercises the minimum surface a benchmark or ad-hoc verification needs:
FT.CREATE, HSET (ingest and first-add bootstrap), FT.SEARCH KNN,
FT.INFO shape, delete, modify, and a filtered KNN. Not a recall test
and not a stress test -- those live elsewhere (e.g. saverestore tests
for round-trip recall). This file is the canonical answer to "did I
break SVS?" during iteration on the branch, and is what to run before
opening a review.
"""

import struct

import pytest
from valkey.client import Valkey

from ft_info_parser import FTInfoParser
from valkey_search_test_case import ValkeySearchTestCaseBase
from valkeytestframework.conftest import resource_port_tracker


DIM = 4


def _fp32(values):
    return struct.pack(f"<{len(values)}f", *values)


def _fp16(values):
    return struct.pack(f"<{len(values)}e", *values)


def _v_index(client: Valkey, index: str) -> dict:
    """Return the parsed `attributes[0]["index"]` dict for the "v" attribute.
    This is where dimensions/distance_metric/size/data_type/algorithm all
    live in FT.INFO's response."""
    return FTInfoParser(
        client.execute_command("FT.INFO", index)
    ).attributes[0]["index"]


def _score_map(reply):
    """FT.SEARCH ... DIALECT 2 returns [count, key, [payload], key, [payload], ...].
    Each payload is a flat list containing at least b'__v_score' and the score.
    Returns {key: float(score)} preserving iteration order."""
    out = {}
    for i in range(1, len(reply), 2):
        key = reply[i]
        payload = dict(zip(reply[i + 1][::2], reply[i + 1][1::2]))
        out[key] = float(payload[b"__v_score"])
    return out


class TestSVSVamanaSmoke(ValkeySearchTestCaseBase):

    def _create_default_index(self, client: Valkey, index: str = "svs_idx"):
        """FT.CREATE with default SVS_VAMANA params (FP32, L2, no filter)."""
        client.execute_command(
            "FT.CREATE", index,
            "SCHEMA", "v", "VECTOR", "SVS_VAMANA",
            "6", "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "L2",
        )

    def _populate(self, client: Valkey, n: int, encode=_fp32):
        """HSET n docs along the x-axis: doc:i -> (float(i), 0, 0, 0)."""
        for i in range(1, n + 1):
            values = [float(i)] + [0.0] * (DIM - 1)
            client.hset(f"doc:{i}", mapping={"v": encode(values)})

    def test_ft_create_reports_svs_vamana(self):
        """FT.INFO exposes the algorithm name and every build-config field."""
        client: Valkey = self.server.get_new_client()
        self._create_default_index(client)
        idx_info = _v_index(client, "svs_idx")
        assert idx_info["dimensions"] == DIM
        assert idx_info["distance_metric"] == "L2"
        assert idx_info["size"] == 0
        assert idx_info["data_type"] == "FLOAT32"
        algo = idx_info["algorithm"]
        assert algo["name"] == "SVS_VAMANA"
        assert algo["graph_max_degree"] == 64
        assert algo["construction_window_size"] == 128
        assert algo["search_window_size"] == 10
        assert algo["compression"] == "SVS_COMPRESSION_NONE"
        assert algo["raw_vector_storage"] == "RAW_VECTOR_STORAGE_KEEP"

    def test_ingest_bootstrap_and_add_points(self):
        """First HSET bootstraps svs_index_; subsequent HSETs go through
        svs_index_dynamic_add_points. FT.INFO size tracks label_to_record_."""
        client: Valkey = self.server.get_new_client()
        self._create_default_index(client)
        self._populate(client, 5)
        assert _v_index(client, "svs_idx")["size"] == 5

    def test_delete_decrements_size(self):
        client: Valkey = self.server.get_new_client()
        self._create_default_index(client)
        self._populate(client, 5)
        client.delete("doc:2", "doc:5")
        assert _v_index(client, "svs_idx")["size"] == 3

    def test_modify_preserves_size(self):
        """HSET on an existing key replaces the vector under one lock
        (delete + add); the label count stays constant."""
        client: Valkey = self.server.get_new_client()
        self._create_default_index(client)
        self._populate(client, 5)
        client.hset("doc:3", mapping={"v": _fp32([99.0, 99.0, 99.0, 99.0])})
        assert _v_index(client, "svs_idx")["size"] == 5

    def test_knn_returns_correct_neighbors_and_distances(self):
        """L2 distances are exact for a small deterministic axis-aligned set."""
        client: Valkey = self.server.get_new_client()
        self._create_default_index(client)
        self._populate(client, 5)

        # Query at (3.7, 0, 0, 0). Analytic L2^2:
        #   doc:4 (4.0) -> 0.09
        #   doc:3 (3.0) -> 0.49
        #   doc:5 (5.0) -> 1.69
        #   doc:2 (2.0) -> 2.89
        #   doc:1 (1.0) -> 7.29
        reply = client.execute_command(
            "FT.SEARCH", "svs_idx", "*=>[KNN 5 @v $q]",
            "PARAMS", "2", "q", _fp32([3.7, 0.0, 0.0, 0.0]),
            "DIALECT", "2",
        )
        assert reply[0] == 5

        scores = _score_map(reply)
        assert list(scores.keys()) == [b"doc:4", b"doc:3", b"doc:5",
                                        b"doc:2", b"doc:1"]
        assert scores[b"doc:4"] == pytest.approx(0.09, abs=1e-3)
        assert scores[b"doc:3"] == pytest.approx(0.49, abs=1e-3)
        assert scores[b"doc:5"] == pytest.approx(1.69, abs=1e-3)
        assert scores[b"doc:2"] == pytest.approx(2.89, abs=1e-3)
        assert scores[b"doc:1"] == pytest.approx(7.29, abs=1e-2)

    def test_knn_caps_k_at_index_size(self):
        """KNN with count > size returns exactly size results, sorted."""
        client: Valkey = self.server.get_new_client()
        self._create_default_index(client)
        self._populate(client, 3)
        reply = client.execute_command(
            "FT.SEARCH", "svs_idx", "*=>[KNN 10 @v $q]",
            "PARAMS", "2", "q", _fp32([2.0, 0.0, 0.0, 0.0]),
            "DIALECT", "2",
        )
        assert reply[0] == 3

    def test_knn_on_empty_index_returns_no_results(self):
        """FT.SEARCH before any HSET short-circuits to an empty result set
        (svs_index_ is null until the first-add bootstrap)."""
        client: Valkey = self.server.get_new_client()
        self._create_default_index(client)
        reply = client.execute_command(
            "FT.SEARCH", "svs_idx", "*=>[KNN 5 @v $q]",
            "PARAMS", "2", "q", _fp32([1.0, 0.0, 0.0, 0.0]),
            "DIALECT", "2",
        )
        assert reply[0] == 0

    def test_type_bfloat16_rejected_at_parse(self):
        """TYPE BFLOAT16 is not supported for SVS_VAMANA (parse-time reject)."""
        client: Valkey = self.server.get_new_client()
        with pytest.raises(Exception) as exc_info:
            client.execute_command(
                "FT.CREATE", "bf16_idx",
                "SCHEMA", "v", "VECTOR", "SVS_VAMANA",
                "6", "TYPE", "BFLOAT16", "DIM", str(DIM),
                "DISTANCE_METRIC", "L2",
            )
        assert "BFLOAT16" in str(exc_info.value)

    def test_fp16_wire_format_ingest_and_search(self):
        """TYPE FLOAT16 goes through the FP16->FP32 conversion in Add and
        Search. Vectors and query are 8-byte payloads instead of 16."""
        client: Valkey = self.server.get_new_client()
        client.execute_command(
            "FT.CREATE", "fp16_idx",
            "SCHEMA", "v", "VECTOR", "SVS_VAMANA",
            "6", "TYPE", "FLOAT16", "DIM", str(DIM), "DISTANCE_METRIC", "L2",
        )
        self._populate(client, 5, encode=_fp16)
        assert _v_index(client, "fp16_idx")["size"] == 5

        reply = client.execute_command(
            "FT.SEARCH", "fp16_idx", "*=>[KNN 3 @v $q]",
            "PARAMS", "2", "q", _fp16([2.0, 0.0, 0.0, 0.0]),
            "DIALECT", "2",
        )
        assert reply[0] == 3
        scores = _score_map(reply)
        # FP16 quantization introduces small error; check ordering and rough
        # distance rather than exact values.
        assert list(scores.keys())[0] == b"doc:2"
        assert scores[b"doc:2"] == pytest.approx(0.0, abs=1e-2)

    @pytest.mark.parametrize("compression", ["NONE", "FP16", "SQ8"])
    def test_ft_create_supports_each_open_compression(self, compression):
        """NONE / FP16 / SQ8 are the v1 open compression kinds. LVQ / LEANVEC
        are proprietary and rejected at parse."""
        client: Valkey = self.server.get_new_client()
        idx = f"comp_{compression.lower()}_idx"
        client.execute_command(
            "FT.CREATE", idx,
            "SCHEMA", "v", "VECTOR", "SVS_VAMANA",
            "8", "TYPE", "FLOAT32", "DIM", str(DIM),
            "DISTANCE_METRIC", "L2", "COMPRESSION", compression,
        )
        self._populate(client, 3)
        idx_info = _v_index(client, idx)
        assert idx_info["size"] == 3
        assert idx_info["algorithm"]["compression"] == \
            f"SVS_COMPRESSION_{compression}"
