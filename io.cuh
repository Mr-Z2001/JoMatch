#ifndef IO_H
#define IO_H

#include "cpuGraph.h"
#include "gpuGraph.h"

void readGraphToCPU(
    cpuGraph *graph,
    const char *filename);

void readGraphToCPU2(
    cpuGraph *graph,
    const char *filename);

void readGraphToCPU_C(
    cpuGraph *graph,
    const char *filename);

void readGraphToCPUDynamic(
    cpuGraph *graph,
    const char *filename);

void copyGraphToGPU(
    gpuGraph *gpuGraph,
    const cpuGraph *cpuGraph,
    bool first_time = true);

void allocateMemGPU(
    gpuGraph *gpuGraph,
    const cpuGraph *cpuGraph,
    bool first_time = true);

void copyGraphToCPU(
    gpuGraph *gpuGraph,
    cpuGraph *cpuGraph);

void copyDataMeta(cpuGraph *data);
void copyMeta(cpuGraph *query, cpuGraph *data);
#endif