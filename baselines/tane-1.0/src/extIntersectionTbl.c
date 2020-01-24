#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef PROFILING
#define MARK
#include <prof.h>
#endif

#include "extIntersectionTbl.h"

static int *tuple_tbl = NULL;
static uint *newtuple_tbl = NULL;
static int *set_tbl = NULL;
static int *set_visit = NULL;
static int noofsets = 0, nooftuples = 0;


void extIntersectionTblINITIALIZE(int argnooftuples)
{
	int i = 0;

	nooftuples = argnooftuples+1;
	noofsets = nooftuples;

	tuple_tbl = (int *) malloc(sizeof(int *)*nooftuples);
	for (i = 0; i < nooftuples; i++) tuple_tbl[i] = 0;

	set_tbl = (int *) malloc(sizeof(int)*noofsets);
	for (i = 0; i < noofsets; i++) set_tbl[i] = 0;

	set_visit = (int *) malloc(sizeof(int)*noofsets);
	for (i = 0; i < noofsets; i++) set_visit[i] = 0;

	newtuple_tbl = (uint *) malloc(sizeof(uint *)*nooftuples);
	for (i = 0; i < nooftuples; i++) newtuple_tbl[i] = 0;
}

void extIntersectionTblLOAD(extPartition *partition)
{
	int i = 0, setno = 1;

	while (i < partition->noofelements) {
	    while (1) {
	        if (partition->elements[i] & endmarker) break;
	        tuple_tbl[(int) (partition->elements[i++])-1] = setno;
	    }
	    tuple_tbl[(int) (partition->elements[i++] & ~endmarker)-1] = setno;
	    setno++;
	}

	noofsets = setno-1;
	set_tbl[noofsets+1] = 1;
}

void extIntersectionTblUNLOAD(extPartition *partition)
{
	int i = 0;

	for (i = 0; i < partition->noofelements; i++)
		tuple_tbl[(partition->elements[i] & ~endmarker)-1] = 0;

	set_tbl[noofsets+1] = 0;
}


#ifndef USEDISK
extPartition *Intersection(extPartition *b)
#else
extPartition *Intersection(extPartition *b, extPartitionFile *file)
#endif
{
	int i = 0, j = 0, k = 0, base_index = 1, setindex = 0;
	uint element = 0;
	extPartition *partition = extPartitionNEW();



	while (i < b->noofelements) {
		setindex = 0;
		while ((b->elements[i] & endmarker) == 0) {
			if (set_tbl[tuple_tbl[b->elements[i++]-1]]++ == 0) {
				set_visit[setindex++] = tuple_tbl[b->elements[i-1]-1];
			}
		}
		b->elements[i] &= ~endmarker;
		if (set_tbl[tuple_tbl[b->elements[i++]-1]]++ == 0) {
			set_visit[setindex++] = tuple_tbl[b->elements[i-1]-1];
		}

		set_tbl[0] = 0;
		for (k = 0; k < setindex; k++) {
			if (set_visit[k] == 0) continue;
			if (set_tbl[set_visit[k]] == 1) {
				set_tbl[set_visit[k]] = 0;
				set_visit[k] = 0;
				continue;
			}
			base_index += set_tbl[set_visit[k]];
			set_tbl[set_visit[k]] = base_index - set_tbl[set_visit[k]];
		}


		for (; j < i; j++) {
			if (set_tbl[tuple_tbl[b->elements[j]-1]] == 0) {
				continue;
			}
			element = set_tbl[tuple_tbl[b->elements[j]-1]]++;
			newtuple_tbl[element-1] = b->elements[j];
			partition->noofelements++;
		}
		b->elements[i-1] |= endmarker;


		for (k = 0; k < setindex; k++) {
			if (set_visit[k] == 0) continue;

			newtuple_tbl[set_tbl[set_visit[k]]-2] |= endmarker;
			partition->noofsets++;

			set_tbl[set_visit[k]] = 0;
		}
	}

#ifndef USEDISK
	partition->elements = (uint *) malloc(sizeof(uint)*partition->noofelements);
	memcpy(partition->elements,newtuple_tbl,sizeof(uint)*partition->noofelements);
#else
  if (partition->noofelements) {
    partition->elements = newtuple_tbl;
    extPartitionFileWRITE(file,partition);
  }
  partition->elements = NULL;
#endif


	return(partition);
}

extPartition *InitialPartition(int *table, int nooftuples)
{
	int i = 0, k = 0, base_index = 1, setindex = 0;
	uint element = 0;
	extPartition *partition = extPartitionNEW();


	for (i = 0; i < nooftuples; i++) set_tbl[i] = 0;


	for (i = 1; i < nooftuples+1; i++) {
		if (set_tbl[table[i-1]]++ == 0) {
			set_visit[setindex++] = table[i-1];
		}
	}

	/*
	printf("set_visit:\n");
	for (k = 0; k < setindex; k++) {
		//printf("[%2d] == %d   set_tbl[%d] == %d\n",k,set_visit[k],
		printf("[%2d] == %d\n",k,set_visit[k]);
	}
	*/


	set_tbl[0] = 0;
	for (k = 0; k < setindex; k++) {
		if (set_visit[k] == 0) continue;
		if (set_tbl[set_visit[k]] == 1) {
			set_tbl[set_visit[k]] = 0;
			set_visit[k] = 0;
			continue;
		}
		base_index += set_tbl[set_visit[k]];
		set_tbl[set_visit[k]] = base_index - set_tbl[set_visit[k]];
	}
	/*
	printf("set_visit:\n");
	for (k = 0; k < setindex; k++) {
		printf("[%2d] == %d   set_tbl[%d] == %d\n",k,set_visit[k],
		set_visit[k],set_tbl[set_visit[k]]);
	}
	*/

	for (i = 1; i < nooftuples+1; i++) {
		if (set_tbl[table[i-1]] == 0) {
			continue;
		}
		element = set_tbl[table[i-1]]++;
		newtuple_tbl[element-1] = i;
		partition->noofelements++;
	}

	/*
	printf("newtuple_tbl:\n");
	for (i = 0; i < partition->noofelements; i++) {
		printf("[%2d] == %d\n",i,newtuple_tbl[i]);
	}
	*/

	memcpy(table,newtuple_tbl,sizeof(uint)*partition->noofelements);

	for (k = 0; k < setindex; k++) {
		if (set_visit[k] == 0) continue;

		table[set_tbl[set_visit[k]]-2] |= endmarker;
		partition->noofsets++;

		set_tbl[set_visit[k]] = 0;
	}


	partition->elements = table;

	return(partition);
}
