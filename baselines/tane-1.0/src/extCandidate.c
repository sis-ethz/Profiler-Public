#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "extCandidate.h"


static int initialized = 0;
static int noofcreated = 0;
static int noofdestroyed = 0;

static void Initialize(void)
{
	noofcreated = 0;
	noofdestroyed = 0;
	initialized = 1;
}


extCandidate *extCandidateNEW(void)
{
	extCandidate *this = NULL;


	if (!initialized) Initialize();


	if ((this = (extCandidate *) malloc(sizeof(extCandidate))) == NULL) {
	   return(NULL);
	}

	this->name = 0;
	this->rhs = 0;
	this->partition = NULL;
	this->next = NULL;
	this->identity = 0;

	noofcreated++;


	return(this);
}

void extCandidateDESTROY(extCandidate *this)
{
	if (this == NULL) return;

	free(this);

	noofdestroyed++;
}

void extCandidateSTATISTICS(void)
{
	printf("No of created extCandidate objects == %d\n",noofcreated);
	printf("\ttotal memory wasted == %d\n",sizeof(extCandidate)*noofcreated);
	printf("No of destroyed extCandidate objects == %d\n",noofdestroyed);
}

void extCandidateDESTROYpartition(extCandidate *this)
{
	extPartitionDESTROY(this->partition);
	this->partition = NULL;
}
