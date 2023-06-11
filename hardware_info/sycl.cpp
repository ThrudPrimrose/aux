#if defined(SYCL)
#include <iostream>
#include <CL/sycl.hpp>
#include <sstream>
#include <algorithm>

/*
      ss << "Device-" << i << ": " << props.name << "\n";
      ss << "  Compute architecture:                       " << props.major << props.minor << "\n";
      ss << "  Global memory:                              " << props.totalGlobalMem / mb << " mb"
         << "\n";
      ss << "  Shared memory:                              " << props.sharedMemPerBlock / kb << " kb"
         << "\n";
      ss << "  Constant memory:                            " << props.totalConstMem / kb << " kb"
         << "\n";
      ss << "  Free memory:                                " << free << " b"
         << " (" << free / mb << " mb)"
         << "\n";
      ss << "  Clock frequency:                            " << props.clockRate / 1000 << " mHz\n";
      ss << "  Compute units (Streaming Multiprocessors):  " << props.multiProcessorCount << "\n";
      ss << "  Warp size:                                  " << props.warpSize << "\n";
      ss << "  Block size:                                 "
         << "<" << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << ">"
         << "\n";
      ss << "  Threads per block:                          " << props.maxThreadsPerBlock << "\n";
      ss << "  Grid size:                                  "
         << "<" << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << ">"
         << "\n";

*/
const std::vector<std::string> properties = {"Device name",
                                             "Global memory",
                                             "Shared memory",
                                             "Free memory",
                                             "Constant memory",
                                             "Clock Frequency",
                                             "Compute Units (Streaming Multiprocessors)",
                                             "Subgroup Size (Warp Size)",
                                             "Work Item Sizes (Block Size)",
                                             "Threads per Work Item",
                                             "Grid Size"};

size_t getPropertyOffset(const char name[])
{
   auto it = std::find(properties.begin(), properties.end(), std::string(name));
   if (it == properties.end())
   {
      throw std::runtime_error("Property input name not a part of the properties list!");
   }
   return std::distance(properties.begin(), it);
}

namespace sycl
{
   void getDeviceInfo(std::vector<std::vector<std::string>> &property_matrix);

   std::string deviceQuery()
   {
      // TODO replace with get_device_number virtual function
      std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices(cl::sycl::info::device_type::gpu);

      unsigned int deviceCount = devices.size() + 1;
      unsigned int propertyCount = properties.size();

      // The matrix is where Column is the list of properties and row is the list of device ids
      std::vector<std::vector<std::string>> propertyMatrix(propertyCount, std::vector<std::string>(deviceCount, "X"));
      getDeviceInfo(propertyMatrix);

      std::stringstream ss;

      // This size is the padding we need to add the porpety-name column
      unsigned int maxPropertyLengthWithPadding = std::max_element(properties.begin(), properties.end(), [](const std::string &lhs, const std::string &rhs)
                                                                   { return lhs.size() < rhs.size(); })
                                                      ->size() +
                                                  1;

      // Get the string entry of a columng with the highest length
      std::vector<unsigned int> columnMaxStringSizes(deviceCount, 0);
      for (unsigned int col = 0; col < deviceCount; col++)
      {
         std::vector<unsigned int> col_sizes;
         for (unsigned int row = 0; row < properties.size(); row++)
         {
            col_sizes.push_back(propertyMatrix[row][col].size());
         }

         // Device Id is also a part of the column
         col_sizes.push_back(std::to_string(col).size());

         unsigned int maxLength = *std::max_element(col_sizes.begin(), col_sizes.end());
         columnMaxStringSizes[col] = maxLength;
      }

      // Generate the first 2 lines with the device ids and dash seperator
      unsigned int lineLength = 0;
      std::string propPadding(maxPropertyLengthWithPadding, ' ');
      ss << propPadding << "|";
      lineLength += maxPropertyLengthWithPadding + 1;
      for (unsigned int i = 0; i < deviceCount; i++)
      {
         std::string paddingNeeded = std::string((columnMaxStringSizes[i] + 1) - std::to_string(i).size(), ' ');
         ss << paddingNeeded << i << "  ";
         lineLength += paddingNeeded.size() + std::to_string(i).size() + 2;
      }
      lineLength -= 2;
      ss.seekp(-2, ss.cur);
      ss << "\n";
      std::string dashLine(lineLength, '-');
      ss << dashLine << "\n";
      // Generation of first 2 lines ends here

      // Iterate the matrix and write per row, also while accessing the padding we need to add
      unsigned int x = 0;
      unsigned int y = 0;
      for (const auto &row : propertyMatrix)
      {
         std::string propPadding(maxPropertyLengthWithPadding - properties[x].size(), ' ');
         ss << properties[x] << propPadding << "|";
         y = 0;
         for (const auto &el : row)
         {
            unsigned int paddingLength = columnMaxStringSizes[y] + 1 - el.size();
            std::string s(paddingLength, ' ');
            ss << s << el << "  ";
            y += 1;
         }
         ss.seekp(-1, ss.cur);
         ss << "\n";
         x += 1;
      }
      std::cout << ss.str() << std::endl;

      return ss.str();
   }
}

namespace sycl
{
   void getDeviceInfo(std::vector<std::vector<std::string>> &propertyMatrix)
   {
      constexpr unsigned int kb = 1024;
      constexpr unsigned int mb = kb * kb;
      std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices(cl::sycl::info::device_type::gpu);

      unsigned int devId = 0;
      for (const auto &dev : devices)
      {
         auto work_item_sizes = dev.get_info<cl::sycl::info::device::max_work_item_sizes<3>>();
         auto subgroup_sizes = dev.get_info<cl::sycl::info::device::sub_group_sizes>();

         std::stringstream ss;
         if (subgroup_sizes.size() > 1)
         {
            ss << "<";
            ss << subgroup_sizes.size();
            for (unsigned int i = 0; i < subgroup_sizes.size() - 1; i++)
            {
               ss << subgroup_sizes[i] << ", ";
            }
            ss << subgroup_sizes[subgroup_sizes.size() - 1] << ">";
         }
         else if (subgroup_sizes.size() == 1)
         {
            ss << subgroup_sizes[0] << "";
         }
         std::string subgroup = ss.str();
         ss.str("");

         if (dev.get_info<cl::sycl::info::device::max_work_item_dimensions>() > 1)
         {
            ss << "<";
            for (unsigned int i = dev.get_info<cl::sycl::info::device::max_work_item_dimensions>() - 1; i > 0; i--)
            {
               ss << work_item_sizes[i] << ", ";
            }
            ss << work_item_sizes[0] << ">";
         }
         else if (dev.get_info<cl::sycl::info::device::max_work_item_dimensions>() == 1)
         {
            ss << work_item_sizes[0] << "";
         }
         std::string work_item = ss.str();
         ss.str("");

         propertyMatrix[getPropertyOffset("Device name")][devId] = dev.get_info<cl::sycl::info::device::name>();
         propertyMatrix[getPropertyOffset("Global memory")][devId] = std::to_string(dev.get_info<cl::sycl::info::device::global_mem_size>() / mb) + " mb";
         propertyMatrix[getPropertyOffset("Shared memory")][devId] = std::to_string(dev.get_info<cl::sycl::info::device::local_mem_size>() / kb) + " kb";
         propertyMatrix[getPropertyOffset("Clock Frequency")][devId] = std::to_string(dev.get_info<cl::sycl::info::device::max_clock_frequency>()) + " mHz";
         propertyMatrix[getPropertyOffset("Compute Units (Streaming Multiprocessors)")][devId] = std::to_string(dev.get_info<cl::sycl::info::device::max_compute_units>());
         propertyMatrix[getPropertyOffset("Subgroup Size (Warp Size)")][devId] = subgroup;
         propertyMatrix[getPropertyOffset("Work Item Sizes (Block Size)")][devId] = work_item;

         devId += 1;
      }
   }
}

int main()
{
   sycl::deviceQuery();
   return 0;
}

#endif