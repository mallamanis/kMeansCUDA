/*
 *  Copyright (C) 2009 by Miltiadis Allamanis
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

/**
 * K-means in Thrust
 *
 */
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/transform.h>
#include <thrust/segmented_scan.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <thrust/utility.h>
#include <thrust/fill.h>
#include <thrust/adjacent_difference.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>

#include <thrust/sorting/radix_sort.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <cstdlib>
#include <stdlib.h>
#include <math.h>

#define BUF_SIZE 1024

typedef thrust::tuple<float,int> IteratorPair;

template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const {
            return x*x;
        }
};

template <typename T>
struct incr
{
    __host__ __device__
        T operator()(const T& x) const {
            return x+1;
        }
};

template <typename T>
struct unitify
{
    __host__ __device__
        T operator()(const T& x) const {
            return x!=0?1:0;
        }
};

struct minPair{
    __host__ __device__
        IteratorPair operator ()(const IteratorPair& a, const IteratorPair& b) const
        {
            if (thrust::get<0>(a) < thrust::get<0>(b))
                return a;
            else
                return b;
        }
};

struct division{
    __host__ __device__
        float operator ()(const float& a, const int& b) const
        {
            if (b==0) return a;
            else return a/b;
        }
};

struct sqDiff{
    __host__ __device__
        float operator ()(const float& a, const int& b) const
        {
            return (a-b)*(a-b);
        }
};

int main(int argc, char** argv)
{
    //Check for wrong parameter count
    if (argc!=3){
        std::cout<<"Error: Wrong number of parameters\n";
        exit(1);
    }

    /*------------------------------------------------------------------
     *  Read input file and store
     *----------------------------------------------------------------*/

    std::cout<<"Loading points from "<<argv[1]<<"...\n";
    unsigned int numAttributes=0,numObjects=0;;
    char line[BUF_SIZE]; //Temporary buffer
    FILE *infile;

    //Open file if possible
    if ((infile = fopen(argv[1], "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", argv[1]);
            exit(1);
    }

    while (fgets(line, BUF_SIZE, infile) != NULL)
            if (strtok(line, " \t\n") != 0)
                numObjects++;

    rewind(infile);
    while (fgets(line, BUF_SIZE, infile) != NULL) {
        if (strtok(line, " \t\n") != 0) {
            /* ignore the id (first attribute): numAttributes = 1; */
            while (strtok(NULL, " ,\t\n") != NULL) numAttributes++;
                break;
        }
   }

    rewind(infile); //Goto the beggining

    thrust::host_vector<float> feature(numObjects*numAttributes);

    int i=0;
     while (fgets(line, BUF_SIZE, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;
            for (unsigned int j=0; j<numAttributes; j++) {
                feature[i]=atof(strtok(NULL, " ,\t\n"));

                i++;
          }
     }

    fclose(infile);  //Close File

    /************************END OF FILE LOADING***********************/


    //Setting number of clusters
    std::cout<<"Using "<<argv[2]<<" clusters.\n";
    int numClusters=atoi(argv[2]);
    thrust::host_vector<float> clusters(numClusters*numAttributes);
    srand(time(NULL));
    //Random Init Clusters
    for (int i=0;i<numClusters;i++){
        int randIndex=rand() % numObjects;
        std::cout<<"Random cluster selected: "<<randIndex<<"\n";
        for (unsigned int j=0;j<numAttributes;j++)
            clusters[i*numAttributes+j]=feature[randIndex*numAttributes+j];
    }

    /*------------------------------------------------------------------
     *  Start k-means
     * ---------------------------------------------------------------*/

    // Create device vectors and transfer to device
    thrust::device_vector<float> dFeature = feature;
    thrust::device_vector<float> dClusters = clusters;

    thrust::device_vector<int> dNumObjects(1);
    dNumObjects[0]=numObjects;

    thrust::device_vector<int> dNumAttributes(1);
    dNumAttributes[0]=(int)numAttributes;

    //Temporary float Vectors
    thrust::device_vector<float> dTemporary2(numObjects);
    thrust::device_vector<float> dTemporary(numObjects*numAttributes);

    thrust::device_vector<float> dResults(numObjects*numClusters);

    //Create Masks and Maps
    thrust::counting_iterator<int> objCount(0);
    thrust::constant_iterator<int> attr(numAttributes);
    thrust::device_vector<int> segScanMask(numObjects*numAttributes);
    thrust::transform(objCount,objCount+numObjects*numAttributes,attr,segScanMask.begin(),thrust::divides<int>());

    thrust::device_vector<int> clusterMask(numObjects*numAttributes);
    thrust::transform(objCount,objCount+numObjects*numAttributes,attr,clusterMask.begin(),thrust::modulus<int>());

    thrust::device_vector<int> gatherDistMask(numObjects);
    thrust::sequence(gatherDistMask.begin(),gatherDistMask.end(),numAttributes-1,numAttributes);

    thrust::device_vector<int> clusterPointsMask(numObjects*numClusters);
    thrust::constant_iterator<int> clust(numClusters);
    thrust::transform(objCount,objCount+numObjects*numClusters,clust,clusterPointsMask.begin(),thrust::modulus<int>());

    thrust::device_vector<int> pointMinMask(numObjects*numClusters);
    thrust::transform(objCount,objCount+numObjects*numClusters,clust,pointMinMask.begin(),thrust::divides<int>());

    thrust::device_vector<int> clusterCoordinate(numClusters*numAttributes);
    thrust::transform(objCount,objCount+numClusters*numAttributes,attr,clusterCoordinate.begin(),thrust::divides<int>());

    thrust::device_vector<int> scatterResults(dNumObjects[0]);

    thrust::device_vector<float> temporaryStorage(dNumObjects[0]);
    thrust::device_vector<float> temporaryStorage2(numClusters);

    thrust::device_vector<int> clusterMapping(numClusters);

    thrust::device_vector<int> divisionBy(numClusters*numAttributes);
    thrust::device_vector<int> objectCount(dNumObjects[0]);
    thrust::device_vector<int> clusterObjectCount(numClusters);


   for (unsigned int iterations=0;iterations<30;iterations++){
        std::cout<<"Iteration "<<iterations<<"...\n";//Start iteration

        thrust::sequence(scatterResults.begin(),scatterResults.end(),0,numClusters);//Caluclate scatterResult

        //For each cluster calculate distances
        for (int i=0;i<numClusters;i++){

            thrust::gather(dTemporary.begin(),dTemporary.end(),clusterMask.begin(),dClusters.begin()+i*dNumAttributes[0]); //Cluster center coordinates
            thrust::transform(dFeature.begin(),dFeature.end(),dTemporary.begin(),dTemporary.begin(),sqDiff()); //Calculate squares

            //Reduce sum per cluster point, using segmented scan...
            thrust::experimental::inclusive_segmented_scan(dTemporary.begin(),dTemporary.end(),segScanMask.begin(),dTemporary.begin())  ;

            thrust::gather(dTemporary2.begin(),dTemporary2.end(),gatherDistMask.begin(),dTemporary.begin()); //Gather all point distances to cluster
            thrust::scatter(dTemporary2.begin(),dTemporary2.end(),scatterResults.begin(),dResults.begin()); //Scatter result to result table
            thrust::transform(scatterResults.begin(),scatterResults.end(),thrust::make_constant_iterator(1),scatterResults.begin(),thrust::plus<int>()); //Calculate next scatterResults


        }//next cluster

        //Find best cluster for each point
            //SegScan for min pair...
            thrust::experimental::inclusive_segmented_scan( thrust::make_zip_iterator(make_tuple(dResults.begin(),clusterPointsMask.begin())),
                                                            thrust::make_zip_iterator(make_tuple(dResults.end(),clusterPointsMask.end())),
                                                            pointMinMask.begin(),
                                                            thrust::make_zip_iterator(make_tuple(dResults.begin(),clusterPointsMask.begin())),
                                                            minPair());

            //Gather clusterNumbers & points...

            thrust::device_vector<int> mapObjects(dNumObjects[0]);
            thrust::sequence(mapObjects.begin(),mapObjects.end(),numClusters-1,numClusters);

            thrust::device_vector<float> pointClusters(dNumObjects[0]);
            thrust::gather(pointClusters.begin(),pointClusters.end(),mapObjects.begin(),clusterPointsMask.begin());

            //Sort by cluster
            thrust::sequence(mapObjects.begin(),mapObjects.end(),0,(int)dNumAttributes[0]); //Create map from point numbers to point coordinate
            thrust::sorting::radix_sort_by_key(pointClusters.begin(),pointClusters.end(),mapObjects.begin());

            //Count points in cluster
                //Segmented Scan
                thrust::fill(objectCount.begin(),objectCount.end(),1);
                thrust::experimental::inclusive_segmented_scan(objectCount.begin(),objectCount.end(),pointClusters.begin(),objectCount.begin());

                //Gather to a vector

                //but first create the gather map
                thrust::device_vector<int> gatherSumsVector(dNumObjects[0]);

                thrust::adjacent_difference(pointClusters.begin(),pointClusters.end(),gatherSumsVector.begin());
                thrust::transform(gatherSumsVector.begin(),gatherSumsVector.end(),gatherSumsVector.begin(),unitify<int>());
                thrust::device_vector<int> activeClusters(1);
                activeClusters[0]=thrust::reduce(gatherSumsVector.begin(),gatherSumsVector.end())+1;
                thrust::transform(gatherSumsVector.begin(),gatherSumsVector.end(),thrust::make_counting_iterator(1),gatherSumsVector.begin(),thrust::multiplies<int>()); //This does not allow clusters with no points...
                thrust::remove(gatherSumsVector.begin(),gatherSumsVector.end(),0);
                //std::cout<<"Active Clusters: "<<activeClusters[0]<<"\n";

                thrust::transform(gatherSumsVector.begin(),gatherSumsVector.end(),thrust::make_constant_iterator(2),gatherSumsVector.begin(),thrust::minus<int>());
                gatherSumsVector[activeClusters[0]-1]=dNumObjects[0]-1; //Last sum will be at the last position

                thrust::device_vector<int> scatterClustersVector(activeClusters[0]);

                thrust::gather(scatterClustersVector.begin(),scatterClustersVector.end(),gatherSumsVector.begin(),pointClusters.begin());

                thrust::device_vector<int> clusterObjectCount(activeClusters[0]);
                thrust::gather(clusterObjectCount.begin(),clusterObjectCount.end(),gatherSumsVector.begin(),objectCount.begin());

            //Create map from point numbers to point coordinate
            //thrust::transform(mapObjects.begin(),mapObjects.end(),thrust::make_constant_iterator(numAttributes),mapObjects.begin(),thrust::multiplies<int>());

            //Create map from cluster coordinate to dClusters
            thrust::transform(scatterClustersVector.begin(),scatterClustersVector.end(),thrust::make_constant_iterator(numAttributes),scatterClustersVector.begin(),thrust::multiplies<int>());

            std::cout<<"Updating clusters...";
            for (unsigned int attrNo=0;attrNo<numAttributes;attrNo++){ //for each attribute (ie coordinate)
                    //gather all point coordinates
                    thrust::gather(temporaryStorage.begin(),temporaryStorage.end(),mapObjects.begin(),dFeature.begin());
                    //segmented scan (sum)
                    thrust::experimental::inclusive_segmented_scan(temporaryStorage.begin(),temporaryStorage.end(),pointClusters.begin(),temporaryStorage.begin());
                    //gather at cluster coordinates
                    thrust::gather(temporaryStorage2.begin(),temporaryStorage2.begin()+activeClusters[0],gatherSumsVector.begin(),temporaryStorage.begin());
                    //scatter to clusters
                    thrust::scatter(temporaryStorage2.begin(),temporaryStorage2.begin()+activeClusters[0],scatterClustersVector.begin(),dClusters.begin()); //Change
                    //increment map to point to next attribute
                    thrust::transform(mapObjects.begin(),mapObjects.end(),thrust::make_constant_iterator(1),mapObjects.begin(),thrust::plus<int>());
                    thrust::transform(scatterClustersVector.begin(),scatterClustersVector.end(),thrust::make_constant_iterator(1),scatterClustersVector.begin(),thrust::plus<int>());
            }

            //for each cluster divide with count by first scattering "count" and then divide-transform.
            thrust::gather(divisionBy.begin(),divisionBy.end(),clusterCoordinate.begin(),clusterObjectCount.begin());
            thrust::transform(dClusters.begin(),dClusters.end(),divisionBy.begin(),dClusters.begin(),division());
    }//next interation


    // Print output
  for (int g=0;g<numClusters;g++){
            std::cout<<"Cluster "<<g<<":(";
            for (int f=0;f<numAttributes;f++)
                std::cout<<dClusters[g*numAttributes+f]<<",";
            std::cout<<")\n";
  }
    return 0;
}



