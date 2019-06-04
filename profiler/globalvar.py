# source type
FILE = "file"
DF   = "df"
DB   = "db"

# embedding types
PRETRAINED_EMBEDDING = "pre-trained",
ONE_HOT_EMBEDDING    = "one-hot"
ATTRIBUTE_EMBEDDING  = "attribute"
TUPLE_EMBEDDING      = "tuple"
EMBEDDING_TYPES      = [PRETRAINED_EMBEDDING, ONE_HOT_EMBEDDING, ATTRIBUTE_EMBEDDING, TUPLE_EMBEDDING]

# data types
NUMERIC     = "numeric"
CATEGORICAL = "categorical"
TEXT        = "text"
DATE        = "date"
DATA_TYPES  = [NUMERIC, CATEGORICAL, TEXT, DATE]

# null policies
NULL_NEQ = "neq"
NULL_EQ  = "eq"
SKIP     = "skip"

# operators
EQ = "equal"
NEQ = "notequal"
GT = "greater_than"
LT = "less_than"
OPERATORS = [EQ, NEQ, GT, LT]

# node types
JOIN = "join"
INTRO = "introduce"
FORGET = "forget"
LEAF = "leaf"
NODE_TYPES = [JOIN, INTRO, FORGET, LEAF]
