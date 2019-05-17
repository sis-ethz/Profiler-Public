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
DATA_TYPES  = [NUMERIC, CATEGORICAL, TEXT]

# null policies
NULL_NEQ = "neq"
NULL_EQ  = "eq"
SKIP     = "skip"
