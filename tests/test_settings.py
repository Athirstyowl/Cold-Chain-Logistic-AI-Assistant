from src.settings import embedding_settings


def test_embedding_mode_is_read_from_the_documented_env_names():
    env = {"EMBEDDING_MODEL": "openai", "LOCAL_EMBEDDING_MODEL": "BAAI/bge-small"}
    assert embedding_settings(env) == ("OPENAI", "BAAI/bge-small")


def test_embedding_defaults_to_local_bge_m3():
    assert embedding_settings({}) == ("LOCAL", "BAAI/bge-m3")


def test_legacy_mixed_case_env_names_are_not_used():
    env = {"Embeddings_model": "OPENAI", "Local_Embedding_Model": "other/model"}
    assert embedding_settings(env) == ("LOCAL", "BAAI/bge-m3")
