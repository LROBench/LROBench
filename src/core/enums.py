from enum import Enum

class OperatorType(Enum):
    MATCH = "match"
    SELECT = "select"
    IMPUTE = "impute"
    CLUSTER = "cluster"
    ORDER = "order"
    MULTI = "multi"

class ImplType(Enum):
    LLM_ALL = "llm_all" # LLM-ALL
    LLM_ONE = "llm_one" # LLM-ONE
    LLM_SEMI = "llm_semi" # LLM-SEMI


class OperandType(Enum):
    CELL = "cell"
    ROW = "row"
    COLUMN = "column"
    TABLE = "table"