from app.schemas.contracts import UtteranceEvent, WordTiming, SpeakerRole
from app.services.stitcher import UtteranceStitcher


def test_stitcher_merge_same_speaker_within_window():
    stitcher = UtteranceStitcher(max_gap_ms=2500)

    utt1 = UtteranceEvent(
        session_id="s1",
        utterance_id="u1",
        speaker=SpeakerRole.CANDIDATE,
        speaker_label="S2",
        text="I used Kafka",
        start_ms=1000,
        end_ms=2000,
        words=[
            WordTiming(word="I", start_ms=1000, end_ms=1200),
            WordTiming(word="used", start_ms=1250, end_ms=1500),
            WordTiming(word="Kafka", start_ms=1550, end_ms=2000),
        ],
    )
    utt2 = UtteranceEvent(
        session_id="s1",
        utterance_id="u2",
        speaker=SpeakerRole.CANDIDATE,
        speaker_label="S2",
        text="for streaming data.",
        start_ms=2800,  # gap is 800ms <= 2500ms
        end_ms=4000,
        words=[
            WordTiming(word="for", start_ms=2800, end_ms=3000),
            WordTiming(word="streaming", start_ms=3050, end_ms=3600),
            WordTiming(word="data.", start_ms=3650, end_ms=4000),
        ],
    )

    flushed1 = stitcher.process_incoming(utt1)
    assert len(flushed1) == 0

    flushed2 = stitcher.process_incoming(utt2)
    assert len(flushed2) == 0

    # Flush session
    final = stitcher.flush_session("s1")
    assert final is not None
    assert final.text == "I used Kafka for streaming data."
    assert final.start_ms == 1000
    assert final.end_ms == 4000
    assert len(final.words) == 6


def test_stitcher_split_on_speaker_change_or_large_gap():
    stitcher = UtteranceStitcher(max_gap_ms=2500)

    # Question from interviewer
    q = UtteranceEvent(
        session_id="s1",
        utterance_id="q1",
        speaker=SpeakerRole.INTERVIEWER,
        speaker_label="S1",
        text="Can you scale Redis?",
        start_ms=0,
        end_ms=2000,
    )
    # Answer from candidate
    a = UtteranceEvent(
        session_id="s1",
        utterance_id="a1",
        speaker=SpeakerRole.CANDIDATE,
        speaker_label="S2",
        text="Yes, with clustering.",
        start_ms=2500,
        end_ms=4000,
    )

    stitcher.process_incoming(q)
    flushed = stitcher.process_incoming(a)

    assert len(flushed) == 1
    assert flushed[0].text == "Can you scale Redis?"
    assert flushed[0].speaker_label == "S1"

    final = stitcher.flush_session("s1")
    assert final.text == "Yes, with clustering."
    assert final.speaker_label == "S2"
