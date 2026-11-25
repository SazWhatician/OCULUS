from typing import Optional, List, Dict
from app.schemas.contracts import UtteranceEvent, WordTiming, SpeakerRole


class UtteranceStitcher:
    """
    Stitches contiguous fragmented utterances from STT streaming chunks.
    Merges if same speaker and gap <= 2500ms; flushes on speaker change or larger gap.
    """

    def __init__(self, max_gap_ms: int = 2500):
        self.max_gap_ms = max_gap_ms
        self._buffers: Dict[str, UtteranceEvent] = {}

    def get_open_utterance(self, session_id: str) -> Optional[UtteranceEvent]:
        return self._buffers.get(session_id)

    def process_incoming(
        self,
        incoming: UtteranceEvent,
        speaker_role_map: Optional[Dict[str, SpeakerRole]] = None,
    ) -> List[UtteranceEvent]:
        """
        Processes an incoming utterance. Returns a list of flushed completed utterances.
        """
        flushed: List[UtteranceEvent] = []
        pending = self._buffers.get(incoming.session_id)

        # Resolve speaker role if mapped (coerce string maps from DB safely)
        if speaker_role_map and incoming.speaker_label in speaker_role_map:
            mapped = speaker_role_map[incoming.speaker_label]
            if isinstance(mapped, SpeakerRole):
                incoming.speaker = mapped
            else:
                try:
                    incoming.speaker = SpeakerRole(str(mapped).strip().lower())
                except ValueError:
                    incoming.speaker = SpeakerRole.UNKNOWN

        if pending is None:
            self._buffers[incoming.session_id] = incoming
            return flushed

        gap_ms = incoming.start_ms - pending.end_ms
        is_same_speaker = pending.speaker_label == incoming.speaker_label

        if is_same_speaker and gap_ms <= self.max_gap_ms:
            # Merge condition
            merged_words = pending.words + incoming.words
            merged_text = f"{pending.text.strip()} {incoming.text.strip()}".strip()
            self._buffers[incoming.session_id] = UtteranceEvent(
                session_id=pending.session_id,
                utterance_id=pending.utterance_id,
                speaker=pending.speaker,
                speaker_label=pending.speaker_label,
                text=merged_text,
                start_ms=pending.start_ms,
                end_ms=max(pending.end_ms, incoming.end_ms),
                words=merged_words,
            )
        else:
            # Flush condition: speaker change or gap > max_gap_ms
            flushed.append(pending)
            self._buffers[incoming.session_id] = incoming

        return flushed

    def flush_session(self, session_id: str) -> Optional[UtteranceEvent]:
        return self._buffers.pop(session_id, None)
