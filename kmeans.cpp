/*
    Programmer:    Manjil M. Pradhan
    K means Algorithm:
    1. Given a set of N training examples each consisting of a vector of continuous attributes, select K training examples to be a set of initial cluster means (centers)
    2. Determine the distance (many options to choose from here) between each training example
       and each of the K cluster means
    3. Assign each training example to the closest cluster mean
    4. Calculate the average of the training examples assigned to each cluster mean (creates a new mean)
    5. Go back to step 2 until the cluster means do not change
       (i.e. all training examples are assigned to the same cluster mean as on the previous iteration)
       
*/

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include <limits>
#include <sstream>
#include <iterator>
#include <set>
//#include "srand-rand-function.h"

using std::vector;
using std::cout;
using std::string;
using std::cin;
using std::endl;
using std::ostream;
using namespace std;

/* typedeffing for ease of use */
typedef vector<double> vd;
typedef vector<vd> vvd;
typedef vector<int> vi;
typedef vector<vector<int > > vvi;

bool check_number(string str);

void printUsage(char* fileName);

void random_shuffle2(vector<string> &trainVect, size_t start, size_t end);

void splitData(ifstream& dataFile, string trainFilename, string testFilename, double split);

void handleFile(const string fn, vvd& v);

// shuffles the traintable and initializes the centroid table with how ever many cluster we have
void initKmeans(vvd& centroidTable, vvd trainTable);

// returns true if centroids have not moved
// calculates the distances between the clusters and the points
bool calculateDistances(const vvd& train, vvd& centroid, vi& clusterTags);

// returns the euclidean distance of a particular vector
double calculateDistance(const vd& train, const vd& center);

// returns true if centroids have not moved
// moves the clusters means based on the number of rows in a cluster (avg)
bool updateClusters(const vvd& trainTable, vvd& centroidTable, const vi& clusterTags);

// comparison of two vectors
bool equal(const vvd& oldCentroids, const vvd& newCentroids);

// goes through the training table and majority votes on which cluster goes with which class label
vi findLabels(const vvd& trainTable, const vi& clusterTags);

// returns how many are right
// tests our assigned labels of test with their actual from the testtable
int testLabels(const vvd& trainTable, const vvd& testTable, const vi& clusterTags, const vi& actualTable);

// returns which row it's in
// finds the closest point from the test table to the train table
int findClosestPoint(const vvd& trainTable, const vd& test);



int NUM_CLUSTERS, NUM_FEATURES, NUM_CLASSIFICATIONS;


int main(int argc, char *argv[])
{
      /*
    * min seed for iris: 1626822009
    * max seed for iris: 1626822104
    *
    * min seed for breast tissue: 1626823074
    * max seed for breast tissue: 1626822789
    *
    */
  string trainFilename, testFilename;
  ifstream data;
  int i, randomSeed;
  
    // string: input training data filename,
    trainFilename = "train";

    // string: input testing data filename
    testFilename = "test";

    //print statement for users to enter the arguments in their command line
    int argCount = argc;
    
    if (argCount == 4)//checking number of argument
      {
    i = 3;
    data.open(argv[i-1]); //takes second from last argument as data file name
    randomSeed = time(NULL);
      }
    else if (argCount == 5)
      {
          i = 4;
          data.open(argv[i-1]); //takes second from last argument as data file name

          if (check_number(argv[1]) && atoi(argv[1]) > 0 && atoi(argv[1]) < RAND_MAX) //checking if seed value given is positive integer and within the range
          {
              randomSeed = atoi(argv[2]); //takes the third value as argument
          }
          else
          {
              cout << "Error in the value of seed. Read the usage and try again.\n";
              printUsage(argv[0]);
              exit(0);
          }
      }
    else
      {
          cout << "Incorrect argument provided in command line. Read the usage and try again.\n";
          printUsage(argv[0]);
          exit(0);
      }

    srand(randomSeed); //i added this line
    
    if (data)
      {
    ifstream data(argv[i-1]);
      }
    else
      {
    cout << "Error while opening file. Check the path and name of the file. Read usage and try again.\n";
    printUsage(argv[0]);
    exit(0);
      }
    
    if (check_number(argv[i]) == true && atoi(argv[i]) > 0) //checking if clusters is int
    {
      NUM_CLUSTERS = atoi(argv[i]);
    }
    else
    {
      cout << "Error while providing value for number of clusters. Read the usage and try again.\n";
      printUsage(argv[0]);
      exit(0);
    }
    
    NUM_FEATURES = 31; //cancer dataset

    string input = argv[1]; //argv[1] = 3578
    cout << "Train Test ration: " << input << "\n";
    int TRAIN_RATIO = stoi(input); //3578
    double split = double(TRAIN_RATIO) / 10;  //train = 357.8


    cout << "RANDOM SEED: " << randomSeed << "\n\n\n";
    splitData(data, trainFilename, testFilename, split);

    //Moving pointer to the begining position of the file.
    data.clear();
    data.seekg(0, ios::beg);


    
    //Moving pointer to the begining position of the file.
    data.clear();
    data.seekg(0, ios::beg);


    int numRight = 0;
    
    vvd trainTable;
    vvd testTable;
    vvd centroidTable;
    
    vi clusterTags;
    vi actualTable;
  
    // read in the training data into the training table
    handleFile(trainFilename, trainTable);
    std::set<int> result;
    for (int i = 0; i < trainTable.size(); ++i)
    {
        result.insert(trainTable[i][NUM_FEATURES - 1]);
    }
    NUM_CLASSIFICATIONS = trainTable.size();
    cout <<"NUM CLASSIFICATION: " << NUM_CLASSIFICATIONS << "\n";

    // same with the testing data
    handleFile(testFilename, testTable);
    NUM_FEATURES = trainTable[0].size();
    initKmeans(centroidTable, trainTable);
    int moves = 0; //iterator
    
    // we need the last cluster tags to help finding and classifying labels
    while (!calculateDistances(trainTable, centroidTable, clusterTags))
    {
        moves++;
    }
    
    cout << "num of moves: " << moves << "\n";
    
    // assign the cluster means to an actual class label
    actualTable = findLabels(trainTable, clusterTags);
    
    // Create output file
    ofstream out("./out.txt");
    
    // returns how many are right
    numRight = testLabels(trainTable, testTable, clusterTags, actualTable);
    int numTotal = testTable.size();
    double accuracy = (float)numRight / (float)numTotal;

    // Send outputs to out file
    cout << "Correct: " << numRight << endl;
    cout << "Total: " << numTotal << endl;
    cout << "Accuracy: " << accuracy << endl;



    return 0;
    
}

bool check_number(string str)
{
  for (int i = 0; i < str.length(); i++)
      if(isdigit(str[i]) == false)
    return false;
      return true;
}

/***************************************************************************************************************************************************
 *This function prints the usage instruction for the user. It shows user how to pass the command line argument while
 *to execute the program.
 ****************************************************************************************************************************************************/

void printUsage(char* fileName)
{
  printf("usage: %10s <unsigned int train-test ratio> [unsigned int seed] <path to data file> <unsigned int Number of Clusters>\n", fileName);
  printf("                  [train-test ratio value range [0-10]] [seed value is less than %d ]\n", RAND_MAX);
}


void random_shuffle2(vector<string> &trainVect, size_t start, size_t end) {
    if (start == end)
    {
        return;
    }
    for (int i = end - 1; i > start + 1; --i) {
        trainVect[i].swap(trainVect[start + rand() % (i + 1)]);
    }
}

void random_shuffle2(vvd& trainTable, size_t start, size_t end) {
    if (start == end)
    {
        return;
    }
    for (int i = end - 1; i > start + 1; --i) {
        trainTable[i].swap(trainTable[start + rand() % (i + 1)]);
    }
}


/****************************************************************************************************************************************************
 *This function uses the split ratio to split the data file into train and test. It saves these data into train and test fIles
 ****************************************************************************************************************************************************/

void splitData(ifstream& dataFile, string trainFilename, string testFilename, double split) {
    ofstream trainFile(trainFilename);
    ofstream testFile(testFilename);

    // Create a list of each line in the data file
    vector<string> lines;
    string line;
    getline(dataFile, line); //i made this change to skip the first line of the data.
    for (; getline(dataFile, line);)
      {
        lines.push_back(line);
      }
    
    random_shuffle2(lines, 0, lines.size()); //why was this commented out?

    // Determine train to test ration
    int split_index = lines.size() * split;
    
    for (size_t i = 0; i < split_index; ++i)
      {
        trainFile << lines[i] << endl;
       
      }

    for (size_t i = split_index; i < lines.size(); ++i)
      {
          testFile << lines[i] << endl;
      }
    
    trainFile.close();
    testFile.close();
}


// handles the file input by opening an input filestream to open a string filename
// then puts the values of doubles into a 2 dimensional vector
void handleFile(const string fn, vvd& v)
{
    vector<double> tmp;
    ifstream in;        // input filestream
    in.open(fn);        // open
    string word;
    for (string line; std::getline(in, line); )
    {
        stringstream curs(line);
        while (std::getline(curs, word, ','))
        {
            tmp.push_back(stod(word));
        }
        v.push_back(tmp);
        tmp.clear();
    }
    
    in.close();                // close
}

// initializes the cluster means by picking a random row from the training table
void initKmeans(vvd& centroidTable, vvd trainTable)
{

    //  good init, but shuffle the train table..
    random_shuffle2(trainTable, 0, trainTable.size());
    cout << "NUM FEATURES : " << NUM_FEATURES << "\n";
    for (int i = 0; i < NUM_CLUSTERS; i++)
    {
        vd tempVec;
        for (int j = 0; j < NUM_FEATURES; j++) {
            tempVec.push_back(trainTable[i][j]);
        }
        centroidTable.push_back(tempVec);
    }
}

// distance function between two vectors
double calculateDistance(const vd& train, const vd& center) {
    double total = 0.0;    // initialize a distance

    // iterate through each column
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        total += pow(center[i] - train[i], 2);    // take the two components, subtract and raise to the second power
    }
    return sqrt(total);    // return the square root of the total
}

// calulates the distances from each cluster mean and their respective points from the training table
bool calculateDistances(const vvd& trainTable, vvd& centroidTable, vi& clusterTags) {

    vd centroid;                // one center information
    vd training;                // one training information
    double dist = 0.0;            // distace calculated between a row in the center and a row in the train table
    double lowestDist = 0.0;    // lowest one in the whole training table
    int lowestCentroid = 0;        // the lowest row number in the centroid table
    int lowestRow = 0;            // the lowest row number in the train table
    bool done = false;            // flag for when to exit the while loop in main
    clusterTags.clear();        // reset the cluster tags

    // for each row in the training table
    for (int r = 0; r < trainTable.size(); r++) {

        lowestDist = std::numeric_limits<double>::max();     // reset distance
        training = trainTable[r];                            // grab the row

        // for each cluster..
        for (int c = 0; c < NUM_CLUSTERS; c++) {
            centroid = centroidTable[c];                    // grab the other row

            // pass the rows to a function that calculates the distance
            dist = calculateDistance(training, centroid);
            //cout << "distance from: [" << r << ", " << c << "] is: " << dist << "\n";

            // if the distance we just got is the lowest in the clusters,
            if (dist < lowestDist)
            {
                lowestDist = dist;            // set some values
                lowestCentroid = c;
                lowestRow = r;
            }
        }
        //cout << "winner is..[" << lowestRow << ", " << lowestCentroid << "] is: " << lowestDist << "\n";
        clusterTags.push_back(lowestCentroid);    // then push the lowest center into a vector of tags

    }
    // update the means of the clusters
    done = updateClusters(trainTable, centroidTable, clusterTags);

    return done;    // return whether the clusters haven't been updated
}

// updates the cluster means by averaging all of the rows that fit the cluster tags from the centroid table
bool updateClusters(const vvd& trainTable, vvd& centroidTable, const vi& clusterTags) {

    vi countedClusters;        // initialize a new vector of ints for the number of clusters in each classification
    bool done = false;        // flag for when to exit the loop in main

    // create a new table for the centroids
    vvd newCentroids(NUM_CLUSTERS, vector<double>(NUM_FEATURES));


    // iterate through the rows
    for (int i = 0; i < trainTable.size(); i++) {
        // and columns of the centroid table
        for (int j = 0; j < centroidTable[0].size(); j++) {
            // increment the new centroid table based on the training table value
            // use the clustertag as the row and the iterator as the column
            newCentroids[clusterTags[i]][j] += trainTable[i][j];
        }
    }

    for (int i = 0; i < NUM_CLUSTERS; i++)
    {
        countedClusters.push_back(count(clusterTags.begin(), clusterTags.end(), i));
    }

    
    //print (countedClusters);
    // divides each value by it's average
    for (int i = 0; i < newCentroids.size(); i++) {
        for (int j = 0; j < newCentroids[0].size(); j++) {

            if (countedClusters[i] != 0)        // prevent nan if the number near the cluster is 0
            {
                newCentroids[i][j] /= countedClusters[i];
            }
            else
            {
                newCentroids[i][j] = centroidTable[i][j];    // use the old one
            }
        }
    }

    // if the number of the old and new cluster are the same, the centroids have not moved
    if (equal(centroidTable, newCentroids)) {
        done = true;
        //cout << "done!\n";        // therefore, set that we are done and return
        return done;
    }

    centroidTable = newCentroids;    // updates the centroid table

    return done;                    // returns false if we get this far
}

// returns whether centroids from old are the same as the new
// if it is, we'll break from the main loop
bool equal(const vvd& oldCentroids, const vvd& newCentroids) {

    size_t rows = newCentroids.size();            // # of rows
    size_t columns = newCentroids[0].size();    // # of columns

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            //if ( fabs( newCentroids[i][j] - oldCentroids[i][j] ) < std::numeric_limits<double>::min() )        // is this close enough?
            // return false;
            //cout << "new " << newCentroids[i][j] << " old " << oldCentroids[i][j] << "\n";
            if (newCentroids[i][j] != oldCentroids[i][j])    // c++11 allows comparison operator
                return false;
        }
    }
    return true;
}

// finds the labels of the training table and returns the actual labels associated with the classificatons
vi findLabels(const vvd& trainTable, const vi& clusterTags)
{
    vvi countTable(NUM_CLUSTERS, vector<int>(NUM_CLASSIFICATIONS));    // temp 2d vector used to count the probability of which classification
    // belongs to a given cluster mean
    vi actualTable(NUM_CLUSTERS);                                // table used to return the actual classification

    int actual = 0, tag = 0; // actual

    // iterate throught the training table..
    for (int r = 0; r < trainTable.size(); r++) {
        actual = trainTable[r][NUM_FEATURES - 1];        // the actual classification for that training set
        tag = clusterTags[r];                        // the tag associated with the training set
        countTable[tag][actual]++;                    // increase the count withe the actual as the row and the tag as the column
    }
    // iterate through the number of cluster means
    for (int r = 0; r < NUM_CLUSTERS; r++) {
        // class label will be assigned to each vector by taking a majority vote amongst it's assigned examples
        // from the training set (ties will be broken by preferring the smallest integer class label).
        actualTable[r] = distance(countTable[r].begin(), max_element(countTable[r].begin(), countTable[r].end()));
    }
    return actualTable;    // return the key map for the actual classifications
}


// tests the labels from the test table
int testLabels(const vvd& trainTable, const vvd& testTable, const vi& clusterTags, const vi& actualTable) {

    int classification = 0, guess = 0, modGuess = 0, correct = 0, tp = 0, fp = 0, tn = 0, fn = 0, positive = 0;
    double precision, recall, f1Score;
    vi guessList{};
    //cout << "oldClassification:\n";
    for (int i = 0; i < testTable.size(); i++)
    {

        // returns the indice of the closest point from the test table set
        // from the entirety of the training table
        classification = findClosestPoint(trainTable, testTable[i]);

        // pick the guess from the position found
        guess = clusterTags[classification];

        // modify the guess from the actual table
        modGuess = actualTable[guess];
        
        // push the modified guess to the guess list
        guessList.push_back(modGuess);
    }
    //cout << "\n";
    //print(guessList);

    // iterate between both sets and count the correct guesses
    for (int i = 0; i < testTable.size(); ++i)
    {
        cout << "guess: " << guessList[i] << " actual: " << testTable[i][NUM_FEATURES - 1] << "\n";
        if (guessList[i] == testTable[i][NUM_FEATURES - 1])
        {
            correct++;    // increase the correct counter if the classification is correct
        }
        
        if (guessList[i] == 0 && testTable[i][NUM_FEATURES - 1] == 0)
            tp++;
        if (guessList[i] == 1 && testTable[i][NUM_FEATURES - 1] == 1)
            tn++;
        if (guessList[i] == 0 && testTable[i][NUM_FEATURES - 1] == 1)
            fp++;
        if (guessList[i] == 1 && testTable[i][NUM_FEATURES - 1] == 0)
            fn++;
        if (testTable[i][NUM_FEATURES - 1] == 0)
            positive++;
    }
    precision = tp / (tp + fp);
    recall = positive / (tp + fn);
    f1Score = 2 * ((precision * recall) / (precision + recall));
    cout << "TP = " << tp << "\n";
    cout << "FP = " << fp << "\n";
    cout << "FN = " << fn << "\n";
    cout << "TN = " << tn << "\n";
    cout << "POSITIVE = " << positive << "\n";
    cout << "PRECISION = " << precision << "\n";
    cout << "RECALL = " << recall << "\n";
    cout << "F-1 Score = " << f1Score << "\n";
    return correct;    // return how many are correct
}


// returns the position of the point (row in traintable) of which the closest point of the test point is
int findClosestPoint(const vvd& trainTable, const vd& test)
{
    double dist = 0.0, closestPoint = 0.0;
    int indice = 0;

    // set the closest point to inf
    closestPoint = std::numeric_limits<double>::max();

    for (int j = 0; j < trainTable.size(); j++) {
        dist = calculateDistance(trainTable[j], test);        // find the distance between the two sets

        // if the distance we just got is the lowest in the clusters,
        if (dist < closestPoint)
        {
            closestPoint = dist;            // set some values
            indice = j;
        }
    }
    //cout << "dist: " << closestPoint << "\n";
    return indice;        // return the indice of the closest point
}


/* Templated Print Statements for double and single vectors */
template <typename t>
void print(vector<vector< t > >  v)
{
    for (auto& i : v) {
        for (auto& j : i) {
            cout << j << " ";
        }
        cout << "\n";
    }cout << "\n";
}


template <typename t>
void print(vector< t >   v)
{
    for (auto& i : v) {
        cout << i << " ";
    }
    cout << "\n";
}

