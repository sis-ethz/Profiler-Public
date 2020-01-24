#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data2int.h"

#define MAX_LINE 500

static void init_dict();
static int get_dict_val(int index, char* str);
static void delete_dict();

void read_data(FILE *file, int **tables, int noofrows, int noofattrs)
{
	char line[MAX_LINE+1];
	char *token;
	int attr, row;

	init_dict(noofattrs);
	for (row=0; row<noofrows; row++) {
		if (fgets(line, MAX_LINE, file) == NULL) {
			fprintf(stderr, "Not enough lines!\n");
			exit(1);
		}
		token = strtok(line, ",");
		for (attr=0; attr<noofattrs; attr++) {
			if (token == NULL) {
				fprintf(stderr, "Not enough attributes on line %d\n", row+1);
				exit(1);
			}
			tables[attr][row] = get_dict_val(attr, token);
			token = strtok(NULL, ",");
		}
	}
	delete_dict();
}



#define MAX_ATTR 63

typedef struct node {
	char ch;
	int newval;
	struct node *next;
	struct node *child;
} node;

static int *nextval;
static node *tries;
static int noofattr = 0;

static void init_dict(int attributes)
{
	int attr;

	noofattr = attributes;
	nextval = (int*) malloc(noofattr*sizeof(int));
	tries = (node*) malloc(noofattr*sizeof(node));
	for (attr=0; attr < noofattr; attr++) {
		nextval[attr] = 1;
		tries[attr].ch = '\0';
		tries[attr].newval = 0;
		tries[attr].next = NULL;
		tries[attr].child = NULL;
	}
}	

static int get_dict_val(int attr, char* str)
{
	node *parent, *curr;
	parent = &tries[attr];
	while (*str) {
		curr = parent->child;
		while (curr != NULL && curr->ch != *str) 
			curr = curr->next;
		if (curr == NULL) break;
		parent = curr;
		++str;
	}
	while (*str) {
		curr = (node*) malloc(sizeof(node));
		curr->ch = *str;
		curr->newval = 0;
		curr->next = parent->child;
		curr->child = NULL;
		parent->child = curr;
		parent = curr;
		++str;
	}
	if (parent->newval == 0) {
		parent->newval = nextval[attr]++;
	}
	return parent->newval;
}

static void delete_subtrie(node *parent)
{
	if (parent != NULL) {
		delete_subtrie(parent->child);
		delete_subtrie(parent->next);
		free(parent);
	}
}

static void delete_dict()
{
	int attr;

	for (attr=0; attr < noofattr; attr++) {
		delete_subtrie(tries[attr].child);
	}
	free(tries);
	free(nextval);
}
