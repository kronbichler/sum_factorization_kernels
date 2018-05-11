
#include <iostream>
#include <iomanip>

#include <mpi.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "evaluation_cell_laplacian_no_vec.h"

#ifdef LIKWID_PERFMON
#include <likwid.h>
#endif

const unsigned int min_degree = 1;
const unsigned int max_degree = 25;

const std::size_t  vector_size_guess = 50000000;
const bool         cartesian         = true;
const unsigned int n_tests           = 10;

typedef double value_type;


template <int dim, int degree, typename Number>
void run_program()
{
  int rank = -1;
  int n_procs = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

  EvaluationCellLaplacianVecEle<dim,degree,Number> evaluator;
  const unsigned int n_cell_batches = std::max(vector_size_guess / Utilities::pow(degree+1,dim) / n_procs,
                                               1UL);
  evaluator.initialize(n_cell_batches, cartesian);

  std::size_t local_size = evaluator.n_elements()*evaluator.dofs_per_cell;
  std::size_t global_size = -1;
  MPI_Allreduce(&local_size, &global_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    {
      std::cout << std::endl;
      std::cout << "Polynomial degree: " << degree << std::endl;
      std::cout << "Vector size:       " << global_size << std::endl;
    }

  evaluator.do_verification();

  double best_avg = std::numeric_limits<double>::max();

  for (unsigned int i=0; i<3; ++i)
    {
      MPI_Barrier(MPI_COMM_WORLD);

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(("cell_laplacian_deg_" + std::to_string(degree)).c_str());
#endif

      struct timeval wall_timer;
      gettimeofday(&wall_timer, NULL);
      double start = wall_timer.tv_sec + 1.e-6 * wall_timer.tv_usec;

      for (unsigned int t=0; t<n_tests; ++t)
        evaluator.matrix_vector_product();

      gettimeofday(&wall_timer, NULL);
      double compute_time = (wall_timer.tv_sec + 1.e-6 * wall_timer.tv_usec - start);

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(("cell_laplacian_deg_" + std::to_string(degree)).c_str());
#endif

      double min_time = -1, max_time = -1, avg_time = -1;
      MPI_Allreduce(&compute_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(&compute_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(&compute_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      best_avg = std::min(best_avg, avg_time/n_procs);
      if (rank == 0)
        {
          std::cout << "Time for operation (min/avg/max): "
                    << min_time/n_tests << " "
                    << avg_time/n_procs/n_tests << " "
                    << max_time/n_tests << " "
                    << std::endl;
        }
    }
  if (rank == 0)
    {
      const std::size_t mem_transfer = global_size * sizeof(Number) *
        (cartesian ? 3 : 13) * n_tests;
      const std::size_t ops_approx = global_size / evaluator.dofs_per_cell
        * (12 * (/*add*/2*((degree+1)/2)*2 +
                 /*mult*/degree+1 +
                 /*fma*/2*((degree-1)*(degree+1)/2))*Utilities::pow(degree+1,dim-1)
           + 3 * (cartesian ? 2 : 5 * 2) * Utilities::pow(degree+1,dim)) * n_tests;
      std::cout << "Degree " << std::setw(2) << degree
                << ", DoFs/s: "
                << global_size * n_tests / best_avg << " with "
                << (double)mem_transfer/best_avg*1e-9 << " GB/s and "
                << (double)ops_approx/best_avg*1e-9 << " GFLOP/s"
                << std::endl;
    }
}


template<int dim, int degree, int max_degree, typename Number>
class RunTime
{
public:
  static void run()
  {
    run_program<dim,degree,Number>();
    RunTime<dim,degree+1,max_degree,Number>::run();
  }
};

template <int dim, int degree,typename Number>
class RunTime<dim,degree,degree,Number>
{
public:
  static void run()
  {
    run_program<dim,degree,Number>();
  }
};

int main(int argc, char** argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#pragma omp parallel
  {
    LIKWID_MARKER_THREADINIT;
  }
#endif

  MPI_Init(&argc, &argv);

  RunTime<3,min_degree,max_degree,value_type>::run();

  MPI_Finalize();

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
