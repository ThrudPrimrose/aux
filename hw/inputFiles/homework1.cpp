// Kerem Mehmet Budanaz 28240
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <cassert>
#include <iomanip>

using namespace std;

// struct for stock
struct Stock
{
	string resource_name;
	int quantity;
};
// struct for consumption
struct Consumption
{
	char building_type;
	std::vector<int> quantity;
};

// Colony can be defined as matrix of char as a typedef

// vector for stock
vector<Stock> mat_stock;
// vector for consumption
vector<Consumption> mat_consumption;
// vector for colony
vector<vector<char>> mat_colony;

int empty_space()
{
	int space = 0;
	for (vector<char> &v : mat_colony)
	{
		for (char c : v)
		{
			if (c == '-')
			{
				space++;
			}
		}
	}
	return space;
}

vector<string> parse_and_split()
{
	string line;
	getline(cin, line);
	stringstream ss(line);
	string word;
	std::vector<string> r;
	while (ss >> word)
	{ // Extract word from the stream.
		r.push_back(std::move(word));
	}
	return r;
}

void terminate(int code)
{
	cout << "Thank you for using the colony management system. The program will terminate." << endl;
	cout << "Goodbye!" << endl;
	exit(code);
}

void print_consumption_per_resource(string s = "")
{
	stringstream ss;
	bool resource_found = false;
	for (int resource_id = 0; resource_id < mat_stock.size(); resource_id++)
	{
		if (s != "" && s != mat_stock[resource_id].resource_name)
		{
			continue;
		}
		resource_found = true;
		ss << "Consumption of resource " << mat_stock[resource_id].resource_name << " by each building in the colony:" << endl;
		for (int i = 0; i < mat_colony.size(); i++)
		{
			for (int j = 0; j < mat_colony[i].size() - 1; j++)
			{
				if (mat_colony[i][j] != '-')
				{
					for (Consumption &cons : mat_consumption)
					{
						if (cons.building_type == mat_colony[i][j])
						{
							ss << cons.quantity[resource_id]
							   << "    ";
							break;
						}
					}
				}
				else
				{
					ss << "0"
					   << "    ";
				}
			}
			if (mat_colony[i][mat_colony[i].size() - 1] != '-')
			{
				for (Consumption &cons : mat_consumption)
				{
					if (cons.building_type == mat_colony[i][mat_colony[i].size() - 1])
					{
						ss << cons.quantity[resource_id]
						   << endl;
						break;
					}
				}
			}
			else
			{
				ss << "0"
				   << endl;
			}
		};
	}
	if (resource_found)
	{
		cout << ss.str();
	}
	else
	{
		cout << "The resource " << s << "was not found in the resources stock." << endl;
	}
}

pair<bool, string> build_initial_colony()
{
	for (vector<char> &v : mat_colony)
	{
		for (char rhs : v)
		{
			if (rhs != '-')
			{
				for (Consumption &consumption : mat_consumption)
				{
					if (consumption.building_type == rhs)
					{
						std::vector<int> used_res = consumption.quantity;
						for (int i = 0; i < mat_stock.size(); i++)
						{
							mat_stock[i].quantity -= used_res[i];
							if (mat_stock[i].quantity < 0)
							{
								return {false, mat_stock[i].resource_name};
							}
						}
					}
				}
			}
		}
	}
	return {true, ""};
}

bool valid_type(char c)
{
	for (Consumption &cons : mat_consumption)
	{
		if (cons.building_type == c)
		{
			return true;
		}
	}
	return false;
}

void build()
{
	int space_left = empty_space();
	if (space_left == 0)
	{
		cout << "There are no empty cells in the colony. Can not add a new building." << endl;
		return;
	}

	cout << "Please enter the type of the building that you want to construct:" << endl;
	string btypestr;
	getline(cin, btypestr);
	char btype = btypestr[0];
	while (btypestr.size() != 1 || !valid_type(btypestr[0]))
	{
		cout << "Invalid building type, please enter a valid building type:" << endl;
		getline(cin, btypestr);
		btype = btypestr[0];
	}

	cout << "Please enter the number of cells that the building will occupy:" << endl;
	string cellcountstr = "-1";
	int cellcount = -1;
	getline(cin, cellcountstr);
	while (true)
	{
		try
		{
			cellcount = std::stoi(cellcountstr);
			if (cellcount > space_left || cellcount <= 0)
			{
				cout << "Invalid number of cells, please enter a valid number of cells:" << endl;
				getline(cin, cellcountstr);
			}
			else
			{
				break;
			}
		}
		catch (std::invalid_argument const &ex)
		{
			cout << "Invalid number of cells, please enter a valid number of cells:" << endl;
			getline(cin, cellcountstr);
		}
	}

	int cur_cell_count = cellcount;
	vector<pair<int, int>> rows_and_cols;
	int i = 1;
	while (cur_cell_count > 0)
	{
		cout << "Please enter the row and the column index of the cell number " << i << ":" << endl;
		int row, col;
		cin >> row >> col;
		while (row < 0 || row >= mat_colony.size() || col < 0 || col >= mat_colony[row].size() || mat_colony[row][col] != '-')
		{
			if (row < 0 || row >= mat_colony.size() || col < 0 || col >= mat_colony[row].size())
			{
				cout << "Invalid row or column index, please enter a valid row and column index:" << endl;
			}
			else if (mat_colony[row][col] != '-')
			{
				cout << "The cell is not empty, please enter the row and the column index of another cell:" << endl;
			}
			cin >> row >> col;
		}
		rows_and_cols.emplace_back(row, col);
		cur_cell_count--;
		i++;
	}

	// Check resources
	for (Consumption &consumption : mat_consumption)
	{
		if (consumption.building_type == btype)
		{
			std::vector<int> used_res = consumption.quantity;
			for (int i = 0; i < mat_stock.size(); i++)
			{
				if (mat_stock[i].quantity < used_res[i] * rows_and_cols.size())
				{
					cout << "Not enough " << mat_stock[i].resource_name << " to build this building" << endl;
					return;
				}
			}
		}
	}

	// Start building, decrease resources
	for (Consumption &consumption : mat_consumption)
	{
		if (consumption.building_type == btype)
		{
			std::vector<int> used_res = consumption.quantity;
			for (int i = 0; i < mat_stock.size(); i++)
			{
				mat_stock[i].quantity -= used_res[i] * rows_and_cols.size();
				assert(mat_stock[i].quantity >= 0);
			}
		}
	}

	for (pair<int, int> rc : rows_and_cols)
	{
		mat_colony[rc.first][rc.second] = btype;
	}

	cout << "The building is added to the colony." << endl;
	return;
}

string stock_to_str()
{
	stringstream ss;
	for (int i = 0; i < mat_stock.size() - 1; i++)
	{
		ss << mat_stock[i].resource_name << " " << mat_stock[i].quantity << endl;
	};
	ss << mat_stock[mat_stock.size() - 1].resource_name
	   << " "
	   << mat_stock[mat_stock.size() - 1].quantity << endl;
	return ss.str();
}

void read_and_print_resource_stock(string &filename, vector<Stock> &mat_stock)
{
	ifstream input;
	Stock stock;
	// We will open stock file
	cout << "Welcome to the colony management system" << endl
		 << "Please enter file name for resources stock:" << endl;
	cin >> filename;
	input.open(filename.c_str());
	while (input.fail())
	{
		cout << "Unable to open the file " << filename << " for reading." << endl
			 << "Please enter the correct file name:" << endl;
		cin >> filename;
		input.open(filename.c_str());
	};
	assert(input.is_open());
	while (!input.eof())
	{
		input >> stock.resource_name >> stock.quantity;
		mat_stock.push_back(stock);
	};

	cout << "Available resources loaded from " << filename << endl
		 << "Resource stock:" << endl;
	cout << stock_to_str();
};

string consumtpion_to_str()
{
	stringstream ss;
	for (int i = 0; i < mat_consumption.size(); i++)
	{
		ss << mat_consumption[i].building_type << " ";
		for (int j = 0; j < mat_consumption[i].quantity.size() - 1; j++)
		{
			ss << mat_consumption[i].quantity[j] << " ";
		}
		ss << mat_consumption[i].quantity[mat_consumption[i].quantity.size() - 1] << endl;
	};
	return ss.str();
}

void read_and_print_consumption(string &filename, vector<Consumption> &mat_consumption)
{
	ifstream input;
	Consumption consumption;
	// We will open consumption file
	cout << "Please enter file name for resource consumption per building type:" << endl;
	cin >> filename;
	input.open(filename.c_str());
	while (input.fail())
	{
		cout << "Unable to open the file " << filename << " for reading." << endl
			 << "Please enter the correct file name:" << endl;
		cin >> filename;
		input.open(filename.c_str());
	};
	assert(!input.fail());
	assert(input.is_open());
	// Read a line and split on empty characters
	string line;
	while (getline(input, line))
	{
		Consumption c;
		// Returns first token
		stringstream ss(line);
		string word;
		ss >> word;
		assert(word.size() == 1);
		c.building_type = word[0];
		while (ss >> word)
		{ // Extract word from the stream.
			c.quantity.push_back(stoi(word));
		}

		// process pair (a,b)
		mat_consumption.push_back(c);
	}

	cout << "Resources consumption per building type loaded from " << filename << endl
		 << "Resources consumption per building type:" << endl;
	cout << consumtpion_to_str();
}

string colony_to_string()
{
	stringstream ss;
	for (int i = 0; i < mat_colony.size(); i++)
	{
		for (int j = 0; j < mat_colony[i].size() - 1; j++)
		{
			ss << mat_colony[i][j] << "    ";
		}
		ss << mat_colony[i][mat_colony[i].size() - 1] << endl;
	};
	return ss.str();
}

bool no_buildings()
{
	for (vector<char> &v : mat_colony)
	{
		for (char c : v)
		{
			if (c != '-')
			{
				return false;
			}
		}
	}
	return true;
}

void destruct()
{
	if (no_buildings())
	{
		cout << "There are no buildings in the colony. Can not remove a building." << endl;
		return;
	}

	cout << "Please enter the row and the column index of the cell that you want to remove:";
	cout << endl;
	int row, col;
	cin >> row >> col;
	while (row < 0 || row >= mat_colony.size() || col < 0 || col >= mat_colony[row].size() || mat_colony[row][col] == '-')
	{
		if (row < 0 || row >= mat_colony.size() || col < 0 || col >= mat_colony[row].size())
		{
			cout << "Invalid row or column index, please enter a valid row and column index:" << endl;
		}
		else if (mat_colony[row][col] == '-')
		{
			cout << "The cell is already empty, please enter the row and the column index of another cell:" << endl;
		}
		cin >> row >> col;
	}
	char c = mat_colony[row][col];
	mat_colony[row][col] = '-';
	for (Consumption &cons : mat_consumption)
	{
		if (cons.building_type == c)
		{
			const std::vector<int> &add_back = cons.quantity;
			for (int i = 0; i < add_back.size(); i++)
			{
				mat_stock[i].quantity += add_back[i];
			}
		}
	}
	cout << "The building is removed, and the corresponding resources are added back to the stock." << endl;
}

void game_loop()
{
	print_consumption_per_resource();
	cout << "Please enter an option number:" << endl;
	cout << "1. Construct a new building on the colony." << endl;
	cout << "2. Destruct/Disassemble a building from the colony." << endl;
	cout << "3. Print the colony." << endl;
	cout << "4. Print the consumption of all resources by each building in the colony." << endl;
	cout << "5. Print the consumption of a specific resource by each building in the colony." << endl;
	cout << "6. Print the resources stock." << endl;
	cout << "7. Exit the program." << endl;
	int i = 0;
	while (true)
	{
		if (i > 0)
		{
			cout << "Please enter an option number:" << endl;
		}

		int option;
		string optionstr;
		getline(cin, optionstr);
		vector<string> l = parse_and_split();
		while (true)
		{
			try
			{
				option = stoi(l[0]);
				break;
			}
			catch (...)
			{
				cout << "Invalid option number.Please enter an option number:";
				cout << endl;

				l = parse_and_split();
			}
		}

		if (option == 1)
		{
			build();
		}
		else if (option == 2)
		{
			destruct();
		}
		else if (option == 3)
		{
			cout << "Colony:" << endl;
			cout << colony_to_string();
		}
		else if (option == 4)
		{
			print_consumption_per_resource();
		}
		else if (option == 5)
		{
			string resource_type;
			cout << "Please enter the type of the resource:" << endl;
			cin >> resource_type;

			bool brk = false;
			for (Stock &st : mat_stock)
			{
				if (st.resource_name == resource_type)
				{
					brk = true;
				}
			}

			if (!brk)
			{
				cout << "The resource " << resource_type << " was not found in the resources stock." << endl;
			}
			else
			{
				print_consumption_per_resource(resource_type);
			}
		}
		else if (option == 6)
		{
			cout << "Resource stock:" << endl;
			cout << stock_to_str();
		}
		else if (option == 7)
		{
			terminate(0);
		}

		i++;
	}
}

void read_and_print_colony(string &filename, vector<vector<char>> &mat_colony)
{
	ifstream input;
	// We will open consumption file
	cout << "Please enter file name for colony:" << endl;
	cin >> filename;
	input.open(filename.c_str());
	while (input.fail())
	{
		cout << "Unable to open the file " << filename << " for reading." << endl
			 << "Please enter the correct file name:" << endl;
		cin >> filename;
		input.open(filename.c_str());
	};
	assert(!input.fail());
	assert(input.is_open());
	// Read a line and split on empty characters
	string line;
	while (getline(input, line))
	{
		Consumption c;
		std::vector<char> colony_line;
		for (char c : line)
		{
			colony_line.push_back(c);
		}
		mat_colony.push_back(colony_line);
	}

	pair<bool, string> enough_resources = build_initial_colony();
	if (!enough_resources.first)
	{
		cout << "Not enough " << enough_resources.second << " to build this building." << endl;
		cout << "Not enough resources to build this colony." << endl;
		terminate(-1);
	}

	cout << "Colony loaded from " << filename << endl
		 << "Colony:" << endl;
	cout << colony_to_string();
	cout << "Resources stock after loading the colony:" << endl;
	cout << "Resource stock:" << endl;
	cout << stock_to_str();
	game_loop();
}

int main()
{
	cout << setw(5) << left;
	string filename;
	ifstream input;
	read_and_print_resource_stock(filename, mat_stock);
	read_and_print_consumption(filename, mat_consumption);
	read_and_print_colony(filename, mat_colony);

	/*Please enter an option number:
	1. Construct a new building on the colony.
	2. Destruct/Disassemble a building from the colony.
	3. Print the colony.
	4. Print the consumption of all resources by each building in the colony.
	5. Print the consumption of a specific resource by each building in the colony.
	6. Print the resources stock.
	7. Exit the program.*/
};