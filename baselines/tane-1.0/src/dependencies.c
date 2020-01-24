#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "extCandidate.h"
#include "extHashTable.h"
#include "extIntersectionTbl.h"
#include "extInputOutput.h"
#include "extPartition.h"
#include "data2int.h"
#include "extG3Tbl.h"

#ifdef USEDISK
#include "extPartitionFile.h"
#endif


#define ATTRIBUTES 32
#define FD(a,b) (a->identity == b->identity)


extCandidate **C = NULL;
extHashTable **C_sub = NULL;
#ifdef USEDISK
extPartitionFile *files[ATTRIBUTES+1];
#endif


int noofattributes = 0, nooftuples = 0, stoplevel = 0;
int noofcandidates[ATTRIBUTES] = { 0 };


#ifdef STATISTICS
int total_no_of_candidates = 0;
int prune_key = 0, prune_key_sub = 0, prune_key_second = 0;
int prune_rhs = 0, prune_rhs_sub = 0, prune_rhs_second = 0;
int noofelements[ATTRIBUTES] = { 0 };
int noofsets[ATTRIBUTES] = { 0 };
#endif

#ifdef APPROXIMATE
int g3_threshold = 0;
#endif

int CalculateFDs(int level);
int compute_rhscand(int level);
int generate_candidates(int level);



int main(int argc, char *argv[])
{
  int index = 0, *tables[ATTRIBUTES];
  extCandidate *candidate = NULL;
  FILE *relation = NULL;
#ifdef APPROXIMATE
  double g3_temp = (double) 0;
#endif


#ifndef APPROXIMATE
  if (argc != 5) {
    printf("Usage: %s stoplevel #tuples #attributes relation\n",argv[0]);
    return(1);
  }
#else
  if (argc != 6) {
    printf("Usage: %s stoplevel #tuples #attributes relation g3_threshold\n",argv[0]);
    return(1);
  }
#endif


  stoplevel = atof(argv[1]);
  nooftuples = atoi(argv[2]);
  noofattributes = atoi(argv[3]);
#ifdef APPROXIMATE
  g3_temp = atof(argv[5]);
  g3_threshold = (int) ((double) nooftuples * g3_temp);
#endif

#ifdef STATISTICS
  printf("\n======================================================================\n");
#ifdef APPROXIMATE
  printf("Parameters (approximate dependencies):\n");
#else
#ifdef USEDISK
  printf("Parameters (disk version):\n");
#else
  printf("Parameters (memory version):\n");
#endif
#endif
//  printf("Parameters:\n");
  printf("No. of tuples            == %d\n",nooftuples);
  printf("No. of attributes        == %d\n",noofattributes);
  printf("Stop level               == %d\n",stoplevel);
  printf("Data                     == %s\n",argv[4]);
#ifdef APPROXIMATE
  printf("Percentage of all tuples == %2.2f %%\n",g3_temp);
  printf("==> G3 threshold         == %d max. rows removed\n",g3_threshold);
#endif
  printf("======================================================================\n\n");
#endif

  if ((relation = fopen(argv[4],"r")) == NULL) {
  	fprintf(stderr,"Error while opening file %s!\n",argv[4]);
  	return(1);
  }

  for (index = 0; index < noofattributes; index++) {
	tables[index] = (int *) malloc(nooftuples*sizeof(int)+1);
  }

  read_data(relation, tables, nooftuples, noofattributes);


  C = (extCandidate **) malloc(sizeof(extCandidate *)*(noofattributes+1));
  C_sub = (extHashTable **) malloc(sizeof(extHashTable *)*(noofattributes+1));

  C[0] = (extCandidate *) malloc(sizeof(extCandidate));
  C[1] = (extCandidate *) malloc(sizeof(extCandidate)*noofattributes);

  C_sub[0] = extHashTableNEW(1);
  C_sub[1] = extHashTableNEW(noofattributes);

  noofcandidates[0] = 1;
  noofcandidates[1] = noofattributes;
#ifdef STATISTICS
  noofelements[0] = 0;
  noofsets[0] = 0;
#endif

  /* Candidate 0 */
  candidate = &C[0][0];
  candidate->name = 0;
  candidate->rhs = 0;
  candidate->partition = extPartitionNEW();
  candidate->partition->noofelements = nooftuples;
  candidate->partition->noofsets = 1;
  candidate->identity = nooftuples-1;
  extBitsetINSERTALL(candidate->rhs,noofattributes);
  extHashTableINSERT(C_sub[0],candidate->name,candidate);

  extIntersectionTblINITIALIZE(nooftuples);

  for (index = 1; index < noofattributes+1; index++) {
    candidate = &C[1][index-1];
    candidate->partition = InitialPartition(tables[index-1],nooftuples);
    candidate->name = 0;
    extBitsetINSERT(candidate->name,index);
    extBitsetINSERTALL(candidate->rhs,noofattributes);
    candidate->identity =
      candidate->partition->noofelements-candidate->partition->noofsets;

    extHashTableINSERT(C_sub[1],candidate->name,candidate);
#ifdef STATISTICS
    noofelements[1] += candidate->partition->noofelements;
    noofsets[1] += candidate->partition->noofsets;
#endif
  }

#ifdef USEDISK
  for (index = 0; index < noofattributes+1; index++) files[index] = NULL;
#endif


#ifndef USEDISK
#ifdef APPROXIMATE
  extG3TblINITIALIZE(nooftuples);
#endif
#endif


  CalculateFDs(1);

#ifdef STATISTICS
  printf("\n======================================================================\n");
  printf("total_no_of_candidates == %d\n",total_no_of_candidates);
  printf("prune_key              == %d\n",prune_key);
  printf("prune_key_sub          == %d\n",prune_key_sub);
  printf("prune_key_second       == %d\n",prune_key_second);
  printf("prune_rhs              == %d\n",prune_rhs);
  printf("prune_rhs_sub          == %d\n",prune_rhs_sub);
  printf("prune_rhs_second       == %d\n",prune_rhs_second);
  printf("======================================================================\n\n");
#endif


  return(0);
}


int CalculateFDs(int level)
{
  while (level < (noofattributes+1) && noofcandidates[level] > 0) {
    fprintf(stderr,"Level == %-2d ",level);
    fprintf(stderr,"#candidates == %-7d ",noofcandidates[level]);
#ifdef STATISTICS
    fprintf(stderr,"avg.elements == %-5d (%d/%d)",
	noofelements[level]/noofsets[level],
	noofelements[level],noofsets[level]);
    total_no_of_candidates += noofcandidates[level];
#endif
    fprintf(stderr,"\n");

    compute_rhscand(level);

    /* cleanup previous levels */
    free(C[level-1]);
    extHashTableDESTROY(C_sub[level-1]);

    if (level >= stoplevel+1) break;

    level = level+1;
    generate_candidates(level);	// includes pruning
  }

  return(0);
}


int compute_rhscand(int level)
{
  int index = 0, element = 0, element1 = 0;
  extCandidate *X = NULL, *sub = NULL;
  extBitset XA = 0, XB = 0, A = 0, R = 0;
#ifdef APPROXIMATE
  int real_g3 = 0;
#endif


  for (index = 0; index < noofcandidates[level]; index++) {
    X = &C[level][index];
    element = extBitsetGETNEXT(X->name,0);
    while (element > 0) {
      XA = X->name; extBitsetREMOVE(XA,element);
      if (extHashTableEXISTS(C_sub[level-1],XA,&sub)) {
        extBitsetINTERSECTION(X->rhs,sub->rhs);
      }

      element = extBitsetGETNEXT(X->name,element);
    }
  }

  for (index = 0; index < noofcandidates[level]; index++) {
    X = &C[level][index];

    A = X->name; extBitsetINTERSECTION(A,X->rhs);

    element = extBitsetGETNEXT(A,0);
    while (element > 0) {
      XA = X->name; extBitsetREMOVE(XA,element);
      if (extHashTableEXISTS(C_sub[level-1],XA,&sub)) {
#ifndef USEDISK
#ifdef APPROXIMATE
        real_g3 = 0;
        if (!FD(sub,X)) {
          real_g3 = sub->identity - X->identity;
          if (real_g3 <= g3_threshold) {
            if (level == 1) {
              real_g3 = nooftuples-extG3BiggestSet(X->partition);
            }
            else {
              extG3TblLOAD(X->partition);
              real_g3 = G3(sub->partition);
              extG3TblUNLOAD(X->partition);
            }
          }
        }
        if (FD(sub,X) || real_g3 <= g3_threshold) {
#else
        if (FD(sub,X)) {
#endif
#else
        if (FD(sub,X)) {
#endif
          R = X->name;
          extBitsetMINUS(R,sub->name);
          extBitsetPRINT(sub->name);
          printf("-> ");
          extBitsetPRINT(R);
#ifndef USEDISK
#ifdef APPROXIMATE
          if (real_g3 != 0) {
            printf("  (%d / %2.2f)", real_g3,(double)real_g3/(double)nooftuples);
          }
#endif
#endif
          printf("\n");

          extBitsetREMOVE(X->rhs,element);
          extBitsetINSERTALL(R,noofattributes);
          extBitsetMINUS(R,X->name);
          extBitsetMINUS(X->rhs,R);

#ifdef APPROXIMATE
          if (real_g3 == 0) {
#else
          {
#endif
            R = extBitsetINSERTALL(R,noofattributes);
            extBitsetMINUS(R,X->name);
            element1 = extBitsetGETNEXT(R,0);
            while (element1 > 0) {
              XB = X->name;
              extBitsetINSERT(XB,element1);
              extBitsetREMOVE(XB,element);
              if (extHashTableEXISTS(C_sub[level],XB,&sub)) {
                extBitsetREMOVE(sub->rhs,element);
              }
              element1 = extBitsetGETNEXT(R,element1);
            }
          }
        }
      }
      element = extBitsetGETNEXT(A,element);
    }


#ifdef APPROXIMATE
    A = X->name;

    element = extBitsetGETNEXT(A,0);
    while (element > 0) {
      XA = X->name; extBitsetREMOVE(XA,element);

      if (extHashTableEXISTS(C_sub[level],XA,&sub)) {
        if (X->identity == sub->identity) {
          R = extBitsetINSERTALL(R,noofattributes);
          extBitsetMINUS(R,X->name);
          element1 = extBitsetGETNEXT(R,0);
          while (element1 > 0) {
            XB = X->name;
            extBitsetINSERT(XB,element1);
            extBitsetREMOVE(XB,element);
            if (extHashTableEXISTS(C_sub[level],XB,&sub)) {
              extBitsetREMOVE(sub->rhs,element);
            }
            element1 = extBitsetGETNEXT(R,element1);
          }
        }
      }
      element = extBitsetGETNEXT(A,element);
    }
#endif

  }

  return(1);
}


int generate_candidates(int level)
{
  int index_x = 0, index_y = 0, prefix = 0, element = 0;
  int loaded = 0, notfound = 0, levelsize = 0;
  extCandidate *X = NULL, *Y = NULL, *candidate = NULL, *sub = NULL;
  extBitset XY = 0, R = 0;
#ifdef USEDISK
  char filename[256];
#endif

  levelsize = (noofcandidates[level-1]*(noofattributes-(level-1)))/level;
  C[level] = (extCandidate *) malloc(sizeof(extCandidate)*levelsize);
  C_sub[level] = extHashTableNEW(levelsize);

#ifdef USEDISK
  sprintf(filename,"/tmp/tane_%d.dat",level);
  files[level] = extPartitionFileNEW();
  extPartitionFileCREATE(files[level],filename);
  extPartitionFileOPEN(files[level-1]);
#endif

  for (index_x = 0; index_x < noofcandidates[level-1]; index_x++) {
    X = &C[level-1][index_x];


    /* pruning keys & full rhs */

    /* key */
#ifdef APPROXIMATE
    if (X->identity <= g3_threshold) {
#else
    if (X->identity == 0) {
#endif
      extBitsetINSERTALL(R,noofattributes);
      extBitsetMINUS(R,X->name);
      extBitsetINTERSECTION(R,X->rhs);

      element = extBitsetGETNEXT(R,0);
      while (element > 0) {
	extBitsetPRINT(X->name);
#ifdef APPROXIMATE
	if (X->identity != 0) {
  printf("-> %d   (approximate key: ",element);
	printf("%d / %2.2f)\n",X->identity,(double)(X->identity)/(double)nooftuples);
//	printf("-> %d   (approximate key == %d)\n",element,X->identity);
	}
	else
	printf("-> %d   (key)\n",element);
#else
	printf("-> %d   (key)\n",element);
#endif
#ifdef APPROXIMATE
        extBitsetREMOVE(X->rhs,element);
#endif
	element = extBitsetGETNEXT(R,element);
      }

#ifdef STATISTICS
      prune_key++;
#endif

#ifdef APPROXIMATE
      if (X->identity == 0) continue;
#else
      continue;
#endif
    }

    /* full rhs */
    if (X->rhs == 0) {
#ifdef STATISTICS
      prune_rhs++;
#endif
      continue;
    }
#ifdef USEDISK
    if (X->partition->elements == NULL) {
      extPartitionFileREAD(files[level-1],X->partition);
    }
#endif


    for (index_y = index_x+1; index_y < noofcandidates[level-1]; index_y++) {
      Y = &C[level-1][index_y];
      prefix = extBitsetPREFIX(X->name,Y->name);
      if (prefix+1 != level-1) break;

#ifdef USEDISK
      if (Y->partition->elements == NULL && Y->partition->noofelements) {
        extPartitionFileREAD(files[level-1],Y->partition);
      }
#endif

      if (Y->identity == 0 || Y->rhs == 0) {
#ifdef STATISTICS
        if (X->identity == 0) prune_key_second++;
	else prune_rhs_second++;
#endif
        continue;
      }


      /* new candidate XY */
      XY = X->name | Y->name;

      notfound = 0;
      element = extBitsetGETNEXT(XY,0);
      while (element > 0) {
	R = XY; extBitsetREMOVE(R,element);
        if (extHashTableEXISTS(C_sub[level-1],R,&sub)) {
	  if (sub->identity == 0 || sub->rhs == 0) {
#ifdef STATISTICS
            if (sub->identity == 0) prune_key_sub++;
	    else prune_rhs_sub++;
#endif
	    notfound = 1;
	    break;
	  }
#ifndef USEDISK
	  if (sub != X && sub != Y) {
	    if (sub->identity < Y->identity) Y = sub;
	  }
#endif
	}
	else notfound = 1;
	element = extBitsetGETNEXT(XY,element);
      }
      if (notfound) continue;

      if (!loaded) {
        extIntersectionTblLOAD(X->partition);
        loaded = 1;
      }


      candidate = &C[level][noofcandidates[level]++];

      candidate->name = XY;

#ifndef USEDISK
      candidate->partition = Intersection(Y->partition);
#else
      candidate->partition = Intersection(Y->partition,files[level]);
#endif

      extBitsetINSERTALL(candidate->rhs,noofattributes);
      candidate->identity =
	candidate->partition->noofelements-candidate->partition->noofsets;

      extHashTableINSERT(C_sub[level],candidate->name,candidate);
#ifdef STATISTICS
      noofelements[level] += candidate->partition->noofelements;
      noofsets[level] += candidate->partition->noofsets;
#endif
    }
    if (loaded) {
      extIntersectionTblUNLOAD(X->partition);
      loaded = 0;
    }
#ifndef APPROXIMATE
    extPartitionDESTROY(X->partition);
    X->partition = NULL;
#endif
  }

#ifdef USEDISK
  extPartitionFileCLOSE(files[level-1]);
  extPartitionFileCLOSE(files[level]);
#endif

  return(1);
}
