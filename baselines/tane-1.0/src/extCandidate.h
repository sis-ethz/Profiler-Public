#ifndef _H_EXTCANDIDATE
#define _H_EXTCANDIDATE	1

#include "extBitset.h"
#include "extPartition.h"


typedef struct extCandidate {
	extBitset		name;
	extBitset		rhs;
	extPartition		*partition;
	struct extCandidate	*next;
	int	identity;
} extCandidate;


extCandidate *extCandidateNEW(void);
void extCandidateDESTROY(extCandidate *this);

void extCandidatePRINT(extCandidate *this);
void extCandidateDESTROYpartition(extCandidate *this);

void extCandidateSTATISTICS(void);

#endif
