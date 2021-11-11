#include <mpi.h>
#include <unordered_map>
#include <tuple>
#include <vector>
#include <iostream>

int calc_offset(int rank, int i) { return (rank > i) ? i : i - 1; }

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank;
  int size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // dont use tuple?
  std::pair<int64_t, int64_t> pr{rank, rank};
  std::vector<MPI_Request> requests;
  requests.resize(size - 1);
  std::vector<std::vector<int64_t>> data;
  data.resize(size - 1);
  std::vector<MPI_Status> recv_stats;
  recv_stats.resize(size - 1);

  for (int i = 0; i < size; i++) {
    if (i == rank) {
      continue;
    }

    MPI_Request req;
    MPI_Isend(&pr.first, 2, MPI_INT64_T, i, i, MPI_COMM_WORLD, &req);
    requests[calc_offset(rank, i)] = req;
    std::cout << 1 << std::endl;
  }

  volatile int got = 0;

  while (got < size - 1) {
    for (int i = 0; i < size; i++) {
      if (i == rank) {
        continue;
      }

      int flag = 0;
      MPI_Status stat;
      MPI_Iprobe(i, rank, MPI_COMM_WORLD, &flag, &stat);
      recv_stats[calc_offset(rank, i)] = stat;

      std::cout << 2 << std::endl;
      if (flag) {
        int count = 0;
        MPI_Get_count(&recv_stats[calc_offset(rank, i)], MPI_INT64_T, &count);

        data[calc_offset(rank, i)].resize(count);
        MPI_Recv(data[calc_offset(rank, i)].data(), count, MPI_INT64_T, i, rank, MPI_COMM_WORLD,
                 &recv_stats[calc_offset(rank, i)]);
        got += 1;
      }
      std::cout << 3 << std::endl;
    }
  }

  std::cout << "uuu" << std::endl;
  MPI_Request* rqs = requests.data();
  MPI_Waitall(requests.size(), rqs, MPI_STATUS_IGNORE);
  std::cout << 4 << std::endl;

  std::cout << rank << " has: ";
  for (auto& el : data) {
    for (auto& t : el) {
      std::cout << t << ", ";
    }
    std::cout << "|";
  }
  std::cout << std::endl;

  std::cout << 5 << std::endl;
  MPI_Finalize();
}