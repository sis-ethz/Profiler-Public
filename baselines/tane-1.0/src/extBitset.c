#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#include "extBitset.h"


int extBitsetGETNEXT(uint bitset, int position)
{
	uint temp = bitset >> position;


	if (temp == 0) return(-1);

	while (1) {
	      position++;
	      if (temp & 1) return(position);
	      temp >>= 1;
	}

	return(-1);
}

void extBitsetPRINT(uint this)
{
	int pos = 0;

	pos = extBitsetGETNEXT(this,0);
	while (pos > -1) {
	      printf("%d ",pos);
	      pos = extBitsetGETNEXT(this,pos);
	}
}

int extBitsetPREFIX(uint a, uint b)
{
	int prefix = 0, pos_a = 0, pos_b = 0;


	pos_a = extBitsetGETNEXT(a,0);
	pos_b = extBitsetGETNEXT(b,0);
	while (pos_a > 0 && pos_b > 0) {
	      if (pos_a != pos_b) break;
	      prefix++;
	      pos_a = extBitsetGETNEXT(a,pos_a);
	      pos_b = extBitsetGETNEXT(b,pos_b);
	}


	return(prefix);
}
