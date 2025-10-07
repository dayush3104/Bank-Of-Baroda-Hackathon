from pydantic import BaseModel
from typing import Literal, List, Tuple

# For the Graph Detector
NodeType = Literal["user", "device", "beneficiary", "ip_address"]

class Node(BaseModel):
    node_id: str
    node_type: NodeType

class Edge(BaseModel):
    source_id: str
    target_id: str

# For the VKYC Verifier
class VerificationPayload(BaseModel):
    challenge_code: str
    spoken_text: str

    head_pose_trace: List[Tuple[float, float]]


class RhythmPayload(BaseModel):
    user_id: str
    timings: List[float] # A list of time intervals between keystrokes in ms
