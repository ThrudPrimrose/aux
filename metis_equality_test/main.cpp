#include <parmetis.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <mpi.h>
#include <cassert>
#include <iostream>
#include <cstdio>

/*

    A grid in this form:

    0 ... (n-1) -> here rank 1
      ...        | m lines

    m*n ... m*n + (n-1)

    ---------------

    ... -> here rank 2

    and then rank 3, rank 4 etc.

*/

constexpr uint64_t lines = 2;
constexpr uint64_t length = 5;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int size;
  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  uint64_t global_offset = rank * lines * length;

  std::vector<idx_t> vtxdist;
  vtxdist.push_back(0);
  std::vector<idx_t> xadj;
  xadj.push_back(0);
  std::vector<idx_t> adjncy;

  for (unsigned int i = 0; i < size; i++) {
    // so we have an array of 0,10k,20k,30k and so on
    vtxdist.push_back(lines * length * (i + 1));
  }

  // each rank has 10 lines that all has 1000 elements each
  // generate xadj array
  if (rank == 0) {
    int sum = 0;
    for (unsigned int i = 0; i < lines; i++) {
      // in the first and last line we have 2,3,...,3,2 neighbors
      // on all other lines we will have 3,4,....,4.3 neighbors
      if (i == 0) {
        sum += 2;
        xadj.push_back(2);
        for (unsigned int j = 1; j < length - 1; j++) {
          sum += 3;
          xadj.push_back(sum);
        }
        sum += 2;
        xadj.push_back(sum);
      } else {
        sum += 3;
        xadj.push_back(sum);
        for (unsigned int j = 1; j < length - 1; j++) {
          sum += 4;
          xadj.push_back(sum);
        }
        sum += 3;
        xadj.push_back(sum);
      }
    }
  } else if (rank < size - 1) {
    int sum = 0;
    for (unsigned int i = 0; i < lines; i++) {
      // in the first and last line we have 2,3,...,3,2 neighbors
      // on all other lines we will have 3,4,....,4.3 neighbors

      sum += 3;
      xadj.push_back(sum);
      for (unsigned int j = 1; j < length - 1; j++) {
        sum += 4;
        xadj.push_back(sum);
      }
      sum += 3;
      xadj.push_back(sum);
    }
  } else {
    int sum = 0;
    for (unsigned int i = 0; i < lines; i++) {
      // in the first and last line we have 2,3,...,3,2 neighbors
      // on all other lines we will have 3,4,....,4.3 neighbors

      if (i == lines - 1) {
        sum += 2;
        xadj.push_back(2);
        for (unsigned int j = 1; j < length - 1; j++) {
          sum += 3;
          xadj.push_back(sum);
        }
        sum += 2;
        xadj.push_back(sum);
      } else {
        sum += 3;
        xadj.push_back(sum);
        for (unsigned int j = 1; j < length - 1; j++) {
          sum += 4;
          xadj.push_back(sum);
        }
        sum += 3;
        xadj.push_back(sum);
      }
    }
  }

  // now generate adjncy
  if (rank == 0) {
    // neighbors are one to the right one to the left one to the up and down
    adjncy.push_back(1);
    adjncy.push_back(length);

    for (unsigned int i = 1; i < length - 1; i++) {
      adjncy.push_back(i - 1);
      adjncy.push_back(i + 1);
      adjncy.push_back(i + length);
    }

    adjncy.push_back(length - 2);
    adjncy.push_back(length + length - 1);

    for (unsigned int i = 1; i < lines; i++) {
      adjncy.push_back(i * length + 1);
      adjncy.push_back(i * length - length);
      adjncy.push_back(i * length + length);

      for (unsigned int j = 1; j < length - 1; j++) {
        adjncy.push_back(i * length + j + 1);
        adjncy.push_back(i * length + j - 1);
        adjncy.push_back(i * length - length + j);
        adjncy.push_back(i * length + length + j);
      }

      adjncy.push_back(i * length + length - 2);
      adjncy.push_back(i * length - length + length - 1);
      adjncy.push_back(i * length + length + length - 1);
    }
  } else if (rank < size - 1) {
    // need to add offset now on
    unsigned int offset = rank * length * lines;

    for (unsigned int i = 0; i < lines; i++) {
      adjncy.push_back(offset + i * length + 1);
      adjncy.push_back(offset + i * length - length);
      adjncy.push_back(offset + i * length + length);

      for (unsigned int j = 1; j < length - 1; j++) {
        adjncy.push_back(offset + i * length + j + 1);
        adjncy.push_back(offset + i * length + j - 1);
        adjncy.push_back(offset + i * length - length + j);
        adjncy.push_back(offset + i * length + length + j);
      }

      adjncy.push_back(offset + i * length + length - 2);
      adjncy.push_back(offset + i * length - length + length - 1);
      adjncy.push_back(offset + i * length + length + length - 1);
    }
  } else {
    unsigned int offset = rank * length * lines;
    // neighbors are one to the right one to the left one to the up and down
    for (unsigned int i = 0; i < lines - 1; i++) {
      adjncy.push_back(offset + i * length + 1);
      adjncy.push_back(offset + i * length - length);
      adjncy.push_back(offset + i * length + length);

      for (unsigned int j = 1; j < length - 1; j++) {
        adjncy.push_back(offset + i * length + j + 1);
        adjncy.push_back(offset + i * length + j - 1);
        adjncy.push_back(offset + i * length - length + j);
        adjncy.push_back(offset + i * length + length + j);
      }

      adjncy.push_back(offset + i * length + length - 2);
      adjncy.push_back(offset + i * length - length + length - 1);
      adjncy.push_back(offset + i * length + length + length - 1);
    }

    unsigned int local_offset = length * (lines - 1);

    adjncy.push_back(offset + local_offset + 1);
    adjncy.push_back(offset + local_offset - length);

    for (unsigned int i = 1; i < length - 1; i++) {
      adjncy.push_back(offset + local_offset + i + 1);
      adjncy.push_back(offset + local_offset + i - 1);
      adjncy.push_back(offset + local_offset + i - length);
    }

    adjncy.push_back(offset + local_offset + length - 2);
    adjncy.push_back(offset + local_offset + length - 1 - length);
  }

  std::vector<idx_t> options{1, 0, 42};
  idx_t wgtflag = 0;
  idx_t numflag = 0;
  idx_t ncon = 1;
  idx_t nparts = size;
  real_t ubvec = 1.05;
  idx_t edgecut = 0;
  std::vector<real_t> tpwgts;
  tpwgts.resize(ncon * size);
  std::fill(tpwgts.begin(), tpwgts.end(), 1.0 / static_cast<float>(size));
  std::vector<idx_t> part;
  part.resize(length * lines);

  std::string filename = std::to_string(rank) + ".out";
  freopen(filename.c_str(), "w", stdout);

  std::cout << "Vtxdistj: {";
  for (auto& el : vtxdist) {
    std::cout << el << ", ";
  }
  std::cout << "}" << std::endl << std::endl;

  std::cout << "Xadj: {";
  for (auto& el : xadj) {
    std::cout << el << ", ";
  }
  std::cout << "}" << std::endl << std::endl;

  std::cout << "Adjncy: {";
  for (auto& el : adjncy) {
    std::cout << el << ", ";
  }
  std::cout << "}" << std::endl << std::endl;

  ParMETIS_V3_PartKway(vtxdist.data(), xadj.data(), adjncy.data(), nullptr, nullptr, &wgtflag,
                       &numflag, &ncon, &nparts, tpwgts.data(), &ubvec, options.data(), &edgecut,
                       part.data(), &comm);

  std::cout << "Part: {";
  for (auto& el : part) {
    std::cout << el << ", ";
  }
  std::cout << "}" << std::endl;

  MPI_Finalize();
}