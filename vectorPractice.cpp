// vectorPractice.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

int max(const std::vector<int>& vec);

int main()
{
    std::vector<int> vec{ 1,2,3,4 };
    std::vector<int>::iterator iter=vec.begin(); //pointer of vector
    std::cout << *iter << std::endl;
    ++iter;
    std::cout << *iter << std::endl;
    ++iter;
    *iter = 9;
    std::cout << vec[3] << std::endl;
    //when dealing with iterator, writing std::vector<int>::iterator is stupid,
    //so auto is useful type def.
    for (auto iter2 = vec.begin(); iter2 != vec.end(); ++iter2)
    {
        std::cout << *iter2 << std::endl;
    }
    
    std::vector<int> data(100);
    for (int i = 0; i < 100; ++i)
        data[i] = rand() % 100; //rand(RAND_MAX) : 0~RAND_MAX
    for (auto iter3 = data.begin(); iter3 != data.end(); ++iter3)
    {
        std::cout << *iter3 << " ";
    }
    std::cout << std::endl;

    std::vector<int> v(10);
    for (int i = 0; i < 10; i++)
    {
        v[i] = rand() % 10;
    }
    //std::cout << "value of 20" << v.at(20) << std::endl; //v.at(20) is safer than v[20]

    int arr[2] = { 1,2 };
    //std::cout << "value of 20 in array" << arr[20] << std::endl; //error in release mode

    //add data to vector
    std::vector<int> v2;
    for (int i = 1; i < 11; i++)
    {
        v2.push_back(i);
    }
    for (auto iter4 = v2.begin(); iter4 != v2.end(); ++iter4)
    {
        std::cout << *iter4 << " ";
    }
    std::cout << std::endl;
    
    std::vector<std::vector<int>> v3;
    for (int i = 1; i < 11; i++)
    {
        std::vector<int> templ{ i,i,i,i };
        v3.push_back(templ);
    }
    
    std::cout << std::endl;
    int length = sizeof(v3) / sizeof(v3[0]);
    std::cout << "length" << length << " elements : " << sizeof(v3[0]) / sizeof(v3[0][0]) << std::endl;
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < sizeof(v3[0]) / sizeof(v3[0][0]); j++)
        {
            std::cout << v3[i][j] << " ";
        }
        std::cout << std::endl;
    }
    //std::cout << v3[length-2][0] << " " << v3[length-3][0]<<" "<<v3[length-4][0] <<std::endl;//<< &(v3.back())-1 << " " << &(v3.back())-2 << std::endl;

    std::vector<int> v6(10, 0);
    v6.insert(v6.begin()+5, 7); //insert at designated position, vector.insert(vector.begin()+idx,value)
    std::cout << v6.at(5) << std::endl;

    std::vector<int> v7(10, 1);
    v7.pop_back();
    std::cout << sizeof(v7) / sizeof(v7[0]) << std::endl;
    
    std::vector<int> v8{ 1,2,3,4,5 };
    v8.erase(v8.begin() + 2); //idx 2 is deleted
    for (auto iter8 = v8.begin(); iter8 != v8.end(); ++iter8)
    {
        std::cout << *iter8 << " ";
    }
    std::cout<<std::endl;
    v8.erase(v8.begin() + 1, v8.begin() + 3); //vector.erase(vector.begin()+idx1,vector.begin()+idx2) : erase from index1 to index2 - 1
    for (auto iter9 = v8.begin(); iter9 != v8.end(); ++iter9)
    {
        std::cout << *iter9 << " ";
    }
    std::cout << std::endl;

    //check vector status
    std::vector<int> v10{1,2,3};
    std::vector<std::vector<int>> v11{ {1,1,1},{2,2,2} };
    if (!v10.empty())
    {
        std::cout << "v10 size : " << v10.size() << std::endl;
        std::cout << "v10 capacity : " << v10.capacity() << std::endl;
        auto first = v10.front();
        auto last = v10.back();
        std::cout << first << " = " << v10[0] << std::endl;
        std::cout << last << " = " << v10[v10.size() - 1] << std::endl;
    }
    if (!v11.empty())
    {
        std::cout << "v11 size : " << v11.size() << std::endl;
        std::cout << "v11 capacity : " << v11.capacity() << std::endl;
        auto first = v11.front().front();
        auto last = v11.back().back();
        int first2 = v11[0][0];
        int last2 = v11[v11.size() - 1][0];
        std::cout << first << " = " << first2 << std::endl;
        std::cout << last << " = " << last2 << std::endl;
        std::cout << "first element address : " << v11.data() << std::endl;
    }

    std::vector<int> v13{ 1,2,3 };
    v13.resize(1);
    std::cout << "resize(1).empty : " << v13.empty() << std::endl;
    std::cout << "resize(1).size() : " << v13.size() << std::endl;
    std::cout << "resize(1).capacity() : " << v13.capacity() << std::endl; //capacity is 3 because initial v13's volume is 3

    std::vector<int> v14{ 1,2,3 };
    v14.reserve(8);
    std::cout << "reserve(8).size()" << v14.size() << std::endl; //size() = 3
    std::cout << "reserve(8).capacity()" << v14.capacity() << std::endl; //capacity() = 8

    std::vector<int> v15{ 1,2,3 };
    v15.resize(1);
    v15.shrink_to_fit();
    std::cout << "shrink_to_fit().empty() : " << v15.empty() << std::endl; //0
    std::cout << "shrink_to_fit().size() : " << v15.size() << std::endl; //1
    std::cout << "shrink_to_fit().capacity() : " << v15.capacity() << std::endl; //1

    std::vector<int> v16{ 1,2,3 };
    std::vector<int> z{ 4,5,6 };
    v16.swap(z);
    for (int i = 0; i < (int)v16.size(); ++i)
    {
        std::cout << v16[i] << " ";
    }
    std::cout << std::endl;

    int dd[][3] = { {1,2,3},{4,5,6} };
    std::vector<std::vector<int>> vv;
    vv.push_back(std::vector<int>(dd[0], std::end(dd[0])));
    vv.push_back(std::vector<int>(dd[1], std::end(dd[1])));

    std::cout << "vv.size() : " << vv.size() << std::endl; //3
    std::cout << "vv[0].size() : " << vv[0].size() << std::endl; //3
    for (int i = 0; i < vv.size(); i++)
        for (int j = 0; j << vv[0].size(); j++)
        {
            std::cout << vv[i][j] << " ";
        }
    std::cout << std::endl;

    std::vector<int> v17{ 1,3,2,5,4,6,3 };
    int mx;
    mx = max(v17);
    std::cout << "max of v17 is " << mx<<std::endl;

    //std::vector function
    std::vector<int> v18{ 1,5,2,6,3,7,3,9 };
    std::cout << "3 in v18 is : " << std::count(v18.begin(), v18.end(), 3) << std::endl;
    auto iter18 = std::find(v18.begin(), v18.end(), 3);
    if (iter18 != v18.end()) //if std::find() can't find designated element, return v18.end() iterator
    {
        std::cout << *iter18 << " is found in " << &(*iter18) <<std::endl;
    }

    std::cout << "sum of v18 is " << std::accumulate(v18.begin(), v18.end(), 0) << std::endl; //std::accumulate(v.begin(),v.end(),0) need #include <numeric>

    std::sort(v18.begin(), v18.end()); //#include <algorithm> is necessary
    std::cout << "sorted vector : ";
    for (auto iter = v18.begin(); iter != v18.end(); ++iter)
    {
        std::cout << *iter << " ";
    }
    std::cout << std::endl;

    std::reverse(v18.begin(), v18.end());
    std::cout << "reversed v18 : ";
    for (auto iter = v18.begin(); iter != v18.end(); ++iter)
    {
        std::cout << *iter << " ";
    }
    std::cout << std::endl;

    std::vector<std::vector<int>> positions{ {0,0,0,0},{0,0,0,0},{0,0,0,0} };
    positions.reserve(5000);
    std::vector<int> pos1{ 1,2,3,4 };
    std::vector<int> pos2{ 5,6,7,8 };
    positions.push_back(pos1);
    positions.push_back(pos2);
    std::cout << "positions size : " << positions.size() << std::endl;
    std::cout << "positions[0].size() : " << positions[0].size() << std::endl;
    std::cout << "{ ";
    for (int i = 0; i < positions.size(); i++)
    {
        std::cout << "{ ";
        for (int j = 0; j < positions[0].size(); j++)
        {
            std::cout << positions[i][j] << " ";
        }
        std::cout << " }" << std::endl;
    }
    std::cout << "}" << std::endl;
        


}

int max(const std::vector<int> &vec)
{
    int max = vec[0];
    for (int i = 1;i < vec.size(); i++)
    {
        if (max < vec[i])
        {
            max = vec[i];
        }
    }
    return max;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
