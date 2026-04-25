from data.personas import PERSONAS
from phase2_content_engine.graph import run_content_engine
from phase2_content_engine.schemas import PostOutput


def test_run_content_engine_schema():
    output = run_content_engine(PERSONAS["BotA"])
    assert isinstance(output, PostOutput)
    assert output.bot_id == "BotA"
    assert output.topic
    assert output.post_content
    assert len(output.post_content) <= 280
