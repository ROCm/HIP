#ifndef RESULT_DATABASE_H
#define RESULT_DATABASE_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cfloat>
using std::string;
using std::vector;
using std::ostream;
using std::ofstream;
using std::ifstream;


// ****************************************************************************
// Class:  ResultDatabase
//
// Purpose:
//   Track numerical results as they are generated.
//   Print statistics of raw results.
//
// Programmer:  Jeremy Meredith
// Creation:    June 12, 2009
//
// Modifications:
//    Jeremy Meredith, Wed Nov 10 14:20:47 EST 2010
//    Split timing reports into detailed and summary.  E.g. for serial code,
//    we might report all trial values, but skip them in parallel.
//
//    Jeremy Meredith, Thu Nov 11 11:40:18 EST 2010
//    Added check for missing value tag.
//
//    Jeremy Meredith, Mon Nov 22 13:37:10 EST 2010
//    Added percentile statistic.
//
//    Jeremy Meredith, Fri Dec  3 16:30:31 EST 2010
//    Added a method to extract a subset of results based on test name.  Also,
//    the Result class is now public, so that clients can use them directly.
//    Added a GetResults method as well, and made several functions const.
//
// ****************************************************************************
class ResultDatabase
{
  public:
    //
    // A performance result for a single SHOC benchmark run.
    //
    struct Result
    {
        string test;  // e.g. "readback"
        string atts;  // e.g. "pagelocked 4k^2"
        string unit;  // e.g. "MB/sec"
        vector<double> value; // e.g. "837.14"
        double GetMin() const;
        double GetMax() const;
        double GetMedian() const;
        double GetPercentile(double q) const;
        double GetMean() const;
        double GetStdDev() const;

        bool operator<(const Result &rhs) const;

        bool HadAnyFLTMAXValues() const
        {
            for (int i=0; i<value.size(); ++i)
            {
                if (value[i] >= FLT_MAX)
                    return true;
            }
            return false;
        }
    };

  protected:
    vector<Result> results;

  public:
    void AddResult(const string &test,
                   const string &atts,
                   const string &unit,
                   double value);
    void AddResults(const string &test,
                    const string &atts,
                    const string &unit,
                    const vector<double> &values);
    vector<Result>        GetResultsForTest(const string &test);
    const vector<Result> &GetResults() const;
    void ClearAllResults();
    void DumpDetailed(ostream&);
    void DumpSummary(ostream&);
    void DumpCsv(string fileName);

  private:
    bool IsFileEmpty(string fileName);

};


#endif
