#include <iostream>
#include <string>
#include <vector>
#include <H5Cpp.h>
#include <atomic>
#include <sstream>
#include <cstdlib>
#include <algorithm>

std::vector<int32_t> read_32(std::string& filename, std::string& dspacename)
{
    std::string fn = filename;
    std::string name = dspacename;

    H5::H5File fp(fn.c_str(),H5F_ACC_RDONLY);
    H5::DataSet dset = fp.openDataSet(name.c_str());

    H5::DataSpace dspace = dset.getSpace();

    // Get the size of the dataset
    hsize_t rank;
    hsize_t dims[1];
    rank = dspace.getSimpleExtentDims(dims); // rank = 1
    //std::cout << "Data size: "<< dims[0] << std::endl; // this is the correct number of values
    //std::cout << "Data rank: "<< rank << std::endl; // this is the correct rank

    // Create memspace
    hsize_t memdim = dims[0];

    // Create storage to hold read data
    std::vector<int32_t> data_out(memdim);

    dset.read(data_out.data(), H5::PredType::NATIVE_INT32, dspace, dspace);

    /*
    for (auto el : data_out)
    {
        std::cout << el <<  " ";
    }
    std::cout << std::endl;
    */

    return data_out;
}

std::vector<int64_t> read_64(std::string& filename, std::string& dspacename)
{
    std::string fn = filename;
    std::string name = dspacename;

    H5::H5File fp(fn.c_str(),H5F_ACC_RDONLY);
    H5::DataSet dset = fp.openDataSet(name.c_str());

    H5::DataSpace dspace = dset.getSpace();

    // Get the size of the dataset
    hsize_t rank;
    hsize_t dims[2];
    rank = dspace.getSimpleExtentDims(dims); // rank = 1
    //std::cout << "Data size: "<< dims[0] << " " << dims[1] << std::endl; // this is the correct number of values
    //std::cout << "Data rank: "<< rank << std::endl; // this is the correct rank

    // Create memspace
    hsize_t memdim = dims[0] * dims[1];

    // Create storage to hold read data
    std::vector<int64_t> data_out(memdim);

    dset.read(data_out.data(), H5::PredType::NATIVE_INT64, dspace, dspace);

    
    /*
    for (auto el : data_out)
    {
        std::cout << el <<  " ";
    }
    std::cout << std::endl;
    */
    

    return data_out;
}

bool neighbors(int64_t v1, int64_t v2, int64_t v3, int64_t v4, int64_t w1, int64_t w2, int64_t w3, int64_t w4)
{
    int acc = 0;
    if (v1 == w1 || v1 == w2 || v1 == w3 || v1 == w4)
    {
        acc += 1;
    }
    if (v2 == w1 || v2 == w2 || v2 == w3 || v2 == w4)
    {
        acc += 1;
    }
    if(acc == 0)
    {
        return false;
    }
    if (v3 == w1 || v3 == w2 || v3 == w3 || v3 == w4)
    {
        acc += 1;
        if (acc == 3)
        {
            return true;
        }
    }
    if(acc <= 1)
    {
        return false;
    }
    if (v4 == w1 || v4 == w2 || v4 == w3 || v4 == w4)
    {
        acc += 1;
    }

    if (acc == 3)
    {
        return true;
    }
    return false;
}

template <typename T>
struct atomicwr
{
  std::atomic<T> _a;

  atomicwr() noexcept
    :_a(0)
  {}

  atomicwr(const std::atomic<T> &a) noexcept
    :_a(a.load())
  {}

  atomicwr(const atomicwr &other) noexcept
    :_a(other._a.load())
  {}

  atomicwr(atomicwr &&other)
    :_a(other._a.load())
  {}

  T load() const noexcept
  {
      return _a.load();
  }

  atomicwr &operator=(const atomicwr &other) noexcept
  {
    _a.store(other._a.load());
  }

  atomicwr &operator=(atomicwr &&other) noexcept
  {
    _a.store(other._a.load());
  }

  bool operator<(const atomicwr& other) const noexcept
  {
      return _a.load() < other.load();
  }

  bool operator>(const atomicwr& other) const noexcept
  {
      return _a.load() > other.load();
  }

  bool operator==(const atomicwr& other) const noexcept
  {
      return _a.load() == other.load();
  }

  atomicwr& operator++() noexcept
  {
    _a++;
    return *this;
  }

atomicwr& operator++(int) noexcept
  {
    _a++;
    return *this;
  }
};


void print_matrix(const std::vector<std::vector<atomicwr<unsigned int>>>& vec)
{
    std::vector<unsigned int> row_max;
    for (const auto& row : vec)
    {
        row_max.push_back(std::max_element(row.begin(), row.end())->load());
    }

    unsigned int glob_max = *std::max_element(row_max.begin(), row_max.end());

    size_t digits = std::to_string(glob_max).size();

    std::stringstream ss;

    ss << "Communication volume: {" << std::endl;
    for (size_t i = 0; i< vec.size(); i++)
    {
        ss << "  {";
        for(size_t j = 0; j < vec[i].size() - 1; j++)
        {
            unsigned int el = vec[i][j].load();
            size_t local_digits = std::to_string(el).size();

            size_t padding = digits - local_digits;

            for(size_t k =0; k<padding+1; k++)
            {
                ss << " ";
            }

            ss << el;
            ss << ",";
        }
            
        unsigned int el = vec[i][vec[i].size() - 1].load();
        size_t local_digits = std::to_string(el).size();

        size_t padding = digits - local_digits;

        for(size_t k =0; k<padding+1; k++)
        {
            ss << " ";
        }

        ss << el;
        ss << "}" << std::endl;
    }
    ss << "}";
    std::cout << ss.str() << std::endl;
}

int main(int argc, char** argv)
{
    std::string fn = argv[1];
    std::cout << fn << std::endl;
    unsigned int rank = std::atoi(argv[2]);
    std::cout << rank << std::endl;
    std::string name = "/mesh0/clustering";
    std::string conn = "/mesh0/connect";
    std::string part = "/mesh0/partition";

    std::vector<std::vector<atomicwr<unsigned int>>> communication_area;

    for(unsigned int i = 0; i < rank; i++)
    {
        std::vector<atomicwr<unsigned int>> r(rank);
        communication_area.push_back(std::move(r));
    }

    // print_matrix(communication_area);

    std::vector<int32_t> vec1 = read_32(fn, name);
    //first 4 elements are of element 1 and 4..8 are from element 2
    std::vector<int64_t> vec2 = read_64(fn, conn);
    std::vector<int32_t> vec3 = read_32(fn, part);

    std::atomic<int> good_cuts = 0;
    std::atomic<int> bad_cuts = 0;
    size_t size = vec2.size()/4;
   
    #pragma omp parallel for schedule(static) firstprivate(vec1, vec2, vec3) shared(good_cuts, bad_cuts)
    for (size_t i = 0; i<size; i++)
    {
        int64_t i4 = 4*i;
        int64_t v1 = vec2[i4];
        int64_t v2 = vec2[i4+1];
        int64_t v3 = vec2[i4+2];
        int64_t v4 = vec2[i4+3];

        int neighbor_count = 0;

        for (size_t j =0; j<size; j++)
        {
            if (i == j)
            {
                continue;
            }

            int64_t j4 = 4*j;
            int64_t w1 = vec2[j4];
            int64_t w2 = vec2[j4+1];
            int64_t w3 = vec2[j4+2];
            int64_t w4 = vec2[j4+3];

            bool neighbor = neighbors(v1,v2,v3,v4,w1,w2,w3,w4);
        
            if (neighbor)
            {
                neighbor_count += 1;

                // std::cout << "(" << v1 << ", " << v2 << ", " << v3 << ", " << v4 << ") and (" << w1 << ", " << w2 << ", " << w3 << ", " << w4 << ") are neighbors" << std::endl; 
                // different rank
                // std::cout << vec3[i] << ", " << vec3[j] << std::endl;
                if (vec3[i] != vec3[j])
                {
                    // same time cluster
                    // std::cout << vec1[i] << ", " << vec1[j] << std::endl;
                    if (vec1[i] == vec1[j])
                    {
                        good_cuts++;
                    }else{
                    // different time cluster
                        bad_cuts++;
                    }
                }

                
                communication_area[vec3[i]][vec3[j]]++;
            }

            if (neighbor_count == 4)
            {
                break;
            }
        }
    }

    std::cout << "good cuts: " << good_cuts << ", " << "bad cuts: " << bad_cuts << std::endl;

    print_matrix(communication_area);
}