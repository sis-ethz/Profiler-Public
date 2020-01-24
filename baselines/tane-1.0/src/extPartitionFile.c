#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#ifdef PROFILING
#define MARK
#include <prof.h>
#endif

#include "extPartitionFile.h"


#ifdef USEDISK

static int noofwrites = 0;

#define BUFFERSIZE	1024
uint filebuffer_write[BUFFERSIZE];
uint filebuffer_read[BUFFERSIZE];


extPartitionFile *extPartitionFileNEW(void)
{
  extPartitionFile *this = NULL;

  if ((this = (extPartitionFile *) malloc(sizeof(extPartitionFile))) == NULL) {
    return(NULL);
  }

  this->fd = NULL;
  this->noofelements = 0;
  this->currentpos = 0;
  this->size = 0;
  this->mode = '\0';
  this->filename[0] = '\0';

  return(this);
}

int extPartitionFileCREATE(extPartitionFile *this, char *filename)
{
  if ((this->fd = fopen(filename,"w")) == NULL) return(-100);

  this->buf_position = 0;

  this->mode = 'w';
  strcpy(this->filename,filename);

  noofwrites = 0;

  return(1);
}

int extPartitionFileOPEN(extPartitionFile *this)
{
  if (this == NULL) return(-10);
  if (this->fd == NULL) return(-100);

  if ((this->fd = fopen(this->filename,"r")) == NULL) return(-1);

  this->buf_position = 0;
  fread(filebuffer_read,sizeof(uint),BUFFERSIZE,this->fd);

  this->mode = 'r';
  this->currentpos = 0;

  return(1);
}

int extPartitionFileCLOSE(extPartitionFile *this)
{
  if (this == NULL) return(-10);

  if (this->mode == 'w') {
    extPartitionFileFLUSH(this);
  }

  fclose(this->fd);
  this->mode = '\0';


  return(1);
}

int extPartitionFileFLUSH(extPartitionFile *this)
{
  fwrite(filebuffer_write,sizeof(uint),this->buf_position,this->fd);
  noofwrites++;

  return(1);
}

int extPartitionFileWRITE(extPartitionFile *this, extPartition *partition)
{
  int j = 0;

  if (this->mode == '\0') return(-100);
  if (this->mode == 'r') return(-101);

  partition->position = this->currentpos;


  for (j = 0; j < partition->noofelements; j++) {
    filebuffer_write[this->buf_position++] = partition->elements[j];
    this->currentpos += 4;
    if (this->buf_position >= BUFFERSIZE) {
      fwrite(filebuffer_write,sizeof(uint),BUFFERSIZE,this->fd);
      this->buf_position = 0;
      noofwrites++;
    }
  }

  this->size = this->currentpos;

#ifdef PRINT
  printf(" writing at position %d; %d elements\n",partition->position,
    partition->noofelements);
#endif

  return(this->currentpos);
}

int extPartitionFileREAD(extPartitionFile *this, extPartition *partition)
{
  int j = 0;
  uint *array = NULL;

  if (this == NULL) return(-10);
  if (this->mode == '\0') return(-100);
  if (this->mode == 'w') return(-101);


#ifdef PRINT
  printf("currentpos == %d\tpartition->position == %d\n",
    this->currentpos,partition->position);
#endif



  this->buf_position += (partition->position-this->currentpos)/4;
  this->currentpos = partition->position;

  if (this->buf_position >= BUFFERSIZE) {
    fseek(this->fd,partition->position,SEEK_SET);
    this->currentpos = ftell(this->fd);
    this->buf_position = 0;
    fread(filebuffer_read,sizeof(uint),BUFFERSIZE,this->fd);
  }

  if ((array = (uint *) malloc(sizeof(uint)*partition->noofelements)) == NULL) {
    return(-1);
  }

  for (j = 0; j < partition->noofelements; j++) {
    array[j] = filebuffer_read[this->buf_position++];
    this->currentpos += 4;
    if (this->buf_position >= BUFFERSIZE) {
      this->buf_position = 0;
      this->currentpos = ftell(this->fd);
      fread(filebuffer_read,sizeof(uint),BUFFERSIZE,this->fd);
    }
  }


#ifdef PRINT
  printf(" reading at position %d; %d elements (array == %d)\n",
    partition->position, partition->noofelements, (int) array);
#endif


  if (partition->elements != NULL) free(partition->elements);
  partition->elements = array;


  return(0);
}

void extPartitionFilePRINT(extPartitionFile *this)
{
}

#endif
