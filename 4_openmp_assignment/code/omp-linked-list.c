#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "omp.h"

const int NUM_NODES = 10;

typedef struct node {
  int data;
  struct node* next;
} node;

node* construct_linked_list(int length) {
  node* head = (node*)malloc(sizeof(node));
  node* current = head;
  node* next;

  // initialize nodes that aren't the last node
  for (int i = 0; i < length - 1; i++) {
    next = (node*)malloc(sizeof(node));
    current->next = next;
    current->data = rand() % 100;
    current = next;
  }

  // initialize the last node
  current->data = rand() % 100;
  current->next = NULL;

  return head;
}

int main() {
  node* current_node = construct_linked_list(NUM_NODES);

  double start = omp_get_wtime();

  #pragma omp parallel
  #pragma omp single
  {
    while (current_node != NULL) {
      #pragma omp task
      {
        // simulating a time-consuming computation
        // sleep for 100,000 microseconds (or 100 milliseconds, or 0.1 seconds)
        usleep(100000);
        printf("Thread %d processed data %d\n",
               omp_get_thread_num(), current_node->data);
      }
      current_node = current_node->next;
    }
  }

  double duration = omp_get_wtime() - start;

  printf("Duration: %.6f\n", duration);
}
