CC=gcc
CFLAGS=-Wall -O4 -DSTATISTICS
OBJECTS=data2int.o extBitset.o extCandidate.o extG3Tbl.o extHashTable.o \
	extInputOutput.o
OBJDISK=$(OBJECTS) dependencies_disk.o extIntersectionTbl_disk.o \
                   extPartitionFile_disk.o extPartition_disk.o
OBJMEM=$(OBJECTS) dependencies_mem.o extIntersectionTbl_mem.o \
                  extPartitionFile_mem.o extPartition_mem.o
OBJG3=$(OBJECTS) dependencies_g3.o extIntersectionTbl_g3.o \
                 extPartitionFile_g3.o extPartition_g3.o
PROGRAMS=tane tanemem taneg3
PRGOBJECTS=tane.o tanemem.o taneg3.o
ALLOBJECTS=$(OBJDISK) $(OBJMEM) $(OBJG3) $(PRGOBJECTS)


all: install

tane: $(OBJDISK)
	$(CC) $(CFLAGS) -o tane $(OBJDISK)

tanemem: $(OBJMEM)
	$(CC) $(CFLAGS) -o tanemem $(OBJMEM)

taneg3: $(OBJG3)
	$(CC) $(CFLAGS) -o taneg3 $(OBJG3)

install: $(PROGRAMS)
	@cp $(PROGRAMS) ../bin

clean:
	@echo "Removing object files..."
	@rm -f $(ALLOBJECTS)


data2int.o: data2int.c data2int.h
dependencies.c: extCandidate.h extBitset.h extPartition.h \
 extHashTable.h extIntersectionTbl.h extInputOutput.h \
 data2int.h extG3Tbl.h
dependencies_disk.o: dependencies.c
	$(CC) $(CFLAGS) -DUSEDISK -c $? -o $@
dependencies_mem.o: dependencies.c
	$(CC) $(CFLAGS) -c $? -o $@
dependencies_g3.o: dependencies.c
	$(CC) $(CFLAGS) -DAPPROXIMATE -c $? -o $@
extBitset.o: extBitset.c extBitset.h
extCandidate.o: extCandidate.c extCandidate.h extBitset.h \
 extPartition.h
extG3Tbl.o: extG3Tbl.c extG3Tbl.h extPartition.h
extHashTable.o: extHashTable.c extHashTable.h extCandidate.h \
 extBitset.h extPartition.h
extInputOutput.o: extInputOutput.c extInputOutput.h extPartition.h
extIntersectionTbl.c: extIntersectionTbl.h extPartition.h
extIntersectionTbl_disk.o: extIntersectionTbl.c
	$(CC) $(CFLAGS) -DUSEDISK -c $? -o $@
extIntersectionTbl_mem.o: extIntersectionTbl.c
	$(CC) $(CFLAGS) -c $? -o $@
extIntersectionTbl_g3.o: extIntersectionTbl.c
	$(CC) $(CFLAGS) -DAPPROXIMATE -c $? -o $@
extPartition.c: extPartition.h
extPartition_disk.o: extPartition.c
	$(CC) $(CFLAGS) -DUSEDISK -c $? -o $@
extPartition_mem.o: extPartition.c
	$(CC) $(CFLAGS) -c $? -o $@
extPartition_g3.o: extPartition.c
	$(CC) $(CFLAGS) -DAPPROXIMATE -c $? -o $@
extPartitionFile.c: extPartitionFile.h extPartition.h
extPartitionFile_disk.o: extPartitionFile.c
	$(CC) $(CFLAGS) -DUSEDISK -c $? -o $@
extPartitionFile_mem.o: extPartitionFile.c
	$(CC) $(CFLAGS) -c $? -o $@
extPartitionFile_g3.o: extPartitionFile.c
	$(CC) $(CFLAGS) -DAPPROXIMATE -c $? -o $@
