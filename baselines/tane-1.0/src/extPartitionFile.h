#ifndef _H_EXTPARTITIONFILE
#define _H_EXTPARTITIONFILE	1


#include "extPartition.h"


typedef struct extPartitionFile {
	FILE	*fd;
	int	noofelements;
	int	currentpos;
	int	size;
	int 	buf_position;
	char	mode;
	char filename[32];
} extPartitionFile;


extPartitionFile *extPartitionFileNEW(void);

int extPartitionFileCREATE(extPartitionFile *this, char *filename);

int extPartitionFileOPEN(extPartitionFile *this);
int extPartitionFileCLOSE(extPartitionFile *this);
int extPartitionFileFLUSH(extPartitionFile *this);

int extPartitionFileREAD(extPartitionFile *this, extPartition *partition);
int extPartitionFileWRITE(extPartitionFile *this, extPartition *partition);

void extPartitionFilePRINT(extPartitionFile *this);


#endif
