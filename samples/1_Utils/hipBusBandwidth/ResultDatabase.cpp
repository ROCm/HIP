#include "ResultDatabase.h"

#include <cfloat>
#include <algorithm>
#include <cmath>
#include <iomanip>

using namespace std;

#define SORT_RETAIN_ATTS_ORDER 1


bool ResultDatabase::Result::operator<(const Result& rhs) const {
    if (test < rhs.test) return true;
    if (test > rhs.test) return false;
#if (SORT_RETAIN_ATTS_ORDER == 0)
    // For ties, sort by the value of the attribute:
    if (atts < rhs.atts) return true;
    if (atts > rhs.atts) return false;
#endif
    return false;  // less-operator returns false on equal
}

double ResultDatabase::Result::GetMin() const {
    double r = FLT_MAX;
    for (int i = 0; i < value.size(); i++) {
        r = min(r, value[i]);
    }
    return r;
}

double ResultDatabase::Result::GetMax() const {
    double r = -FLT_MAX;
    for (int i = 0; i < value.size(); i++) {
        r = max(r, value[i]);
    }
    return r;
}

double ResultDatabase::Result::GetMedian() const { return GetPercentile(50); }

double ResultDatabase::Result::GetPercentile(double q) const {
    int n = value.size();
    if (n == 0) return FLT_MAX;
    if (n == 1) return value[0];

    if (q <= 0) return value[0];
    if (q >= 100) return value[n - 1];

    double index = ((n + 1.) * q / 100.) - 1;

    vector<double> sorted = value;
    sort(sorted.begin(), sorted.end());

    if (n == 2) return (sorted[0] * (1 - q / 100.) + sorted[1] * (q / 100.));

    int index_lo = int(index);
    double frac = index - index_lo;
    if (frac == 0) return sorted[index_lo];

    double lo = sorted[index_lo];
    double hi = sorted[index_lo + 1];
    return lo + (hi - lo) * frac;
}

double ResultDatabase::Result::GetMean() const {
    double r = 0;
    for (int i = 0; i < value.size(); i++) {
        r += value[i];
    }
    return r / double(value.size());
}

double ResultDatabase::Result::GetStdDev() const {
    double r = 0;
    double u = GetMean();
    if (u == FLT_MAX) return FLT_MAX;
    for (int i = 0; i < value.size(); i++) {
        r += (value[i] - u) * (value[i] - u);
    }
    r = sqrt(r / value.size());
    return r;
}


void ResultDatabase::AddResults(const string& test, const string& atts, const string& unit,
                                const vector<double>& values) {
    for (int i = 0; i < values.size(); i++) {
        AddResult(test, atts, unit, values[i]);
    }
}

static string RemoveAllButLeadingSpaces(const string& a) {
    string b;
    int n = a.length();
    int i = 0;
    while (i < n && a[i] == ' ') {
        b += a[i];
        ++i;
    }
    for (; i < n; i++) {
        if (a[i] != ' ' && a[i] != '\t') b += a[i];
    }
    return b;
}

void ResultDatabase::AddResult(const string& test_orig, const string& atts_orig,
                               const string& unit_orig, double value) {
    string test = RemoveAllButLeadingSpaces(test_orig);
    string atts = RemoveAllButLeadingSpaces(atts_orig);
    string unit = RemoveAllButLeadingSpaces(unit_orig);
    int index;
    for (index = 0; index < results.size(); index++) {
        if (results[index].test == test && results[index].atts == atts) {
            if (results[index].unit != unit) throw "Internal error: mixed units";

            break;
        }
    }

    if (index >= results.size()) {
        Result r;
        r.test = test;
        r.atts = atts;
        r.unit = unit;
        results.push_back(r);
    }

    results[index].value.push_back(value);
}

// ****************************************************************************
//  Method:  ResultDatabase::DumpDetailed
//
//  Purpose:
//    Writes the full results, including all trials.
//
//  Arguments:
//    out        where to print
//
//  Programmer:  Jeremy Meredith
//  Creation:    August 14, 2009
//
//  Modifications:
//    Jeremy Meredith, Wed Nov 10 14:25:17 EST 2010
//    Renamed to DumpDetailed to make room for a DumpSummary.
//
//    Jeremy Meredith, Thu Nov 11 11:39:57 EST 2010
//    Added note about (*) missing value tag.
//
//    Jeremy Meredith, Tue Nov 23 13:57:02 EST 2010
//    Changed note about missing values to be worded a little better.
//
// ****************************************************************************
void ResultDatabase::DumpDetailed(ostream& out) {
    vector<Result> sorted(results);

    stable_sort(sorted.begin(), sorted.end());

    const int testNameW = 24;
    const int attW = 12;
    const int fieldW = 11;
    out << std::fixed << right << std::setprecision(4);

    int maxtrials = 1;
    for (int i = 0; i < sorted.size(); i++) {
        if (sorted[i].value.size() > maxtrials) maxtrials = sorted[i].value.size();
    }

    // TODO: in big parallel runs, the "trials" are the procs
    // and we really don't want to print them all out....
    out << setw(testNameW) << "test\t" << setw(attW) << "atts\t" << setw(fieldW) << "median\t"
        << "mean\t"
        << "stddev\t"
        << "min\t"
        << "max\t";
    for (int i = 0; i < maxtrials; i++) out << "trial" << i << "\t";
    out << endl;

    for (int i = 0; i < sorted.size(); i++) {
        Result& r = sorted[i];
        out << setw(testNameW) << r.test + "\t";
        out << setw(attW) << r.atts + "\t";
        out << setw(fieldW) << r.unit + "\t";
        if (r.GetMedian() == FLT_MAX)
            out << "N/A\t";
        else
            out << r.GetMedian() << "\t";
        if (r.GetMean() == FLT_MAX)
            out << "N/A\t";
        else
            out << r.GetMean() << "\t";
        if (r.GetStdDev() == FLT_MAX)
            out << "N/A\t";
        else
            out << r.GetStdDev() << "\t";
        if (r.GetMin() == FLT_MAX)
            out << "N/A\t";
        else
            out << r.GetMin() << "\t";
        if (r.GetMax() == FLT_MAX)
            out << "N/A\t";
        else
            out << r.GetMax() << "\t";
        for (int j = 0; j < r.value.size(); j++) {
            if (r.value[j] == FLT_MAX)
                out << "N/A\t";
            else
                out << r.value[j] << "\t";
        }

        out << endl;
    }
    out << endl
        << "Note: Any results marked with (*) had missing values." << endl
        << "      This can occur on systems with a mixture of" << endl
        << "      device types or architectural capabilities." << endl;
}


// ****************************************************************************
//  Method:  ResultDatabase::DumpDetailed
//
//  Purpose:
//    Writes the summary results (min/max/stddev/med/mean), but not
//    every individual trial.
//
//  Arguments:
//    out        where to print
//
//  Programmer:  Jeremy Meredith
//  Creation:    November 10, 2010
//
//  Modifications:
//    Jeremy Meredith, Thu Nov 11 11:39:57 EST 2010
//    Added note about (*) missing value tag.
//
// ****************************************************************************
void ResultDatabase::DumpSummary(ostream& out) {
    vector<Result> sorted(results);

    stable_sort(sorted.begin(), sorted.end());

    const int testNameW = 24;
    const int attW = 12;
    const int fieldW = 9;
    out << std::fixed << right << std::setprecision(4);

    // TODO: in big parallel runs, the "trials" are the procs
    // and we really don't want to print them all out....
    out << setw(testNameW) << "test\t" << setw(attW) << "atts\t" << setw(fieldW) << "units\t"
        << "median\t"
        << "mean\t"
        << "stddev\t"
        << "min\t"
        << "max\t";
    out << endl;

    for (int i = 0; i < sorted.size(); i++) {
        Result& r = sorted[i];
        out << setw(testNameW) << r.test + "\t";
        out << setw(attW) << r.atts + "\t";
        out << setw(fieldW) << r.unit + "\t";
        if (r.GetMedian() == FLT_MAX)
            out << "N/A\t";
        else
            out << r.GetMedian() << "\t";
        if (r.GetMean() == FLT_MAX)
            out << "N/A\t";
        else
            out << r.GetMean() << "\t";
        if (r.GetStdDev() == FLT_MAX)
            out << "N/A\t";
        else
            out << r.GetStdDev() << "\t";
        if (r.GetMin() == FLT_MAX)
            out << "N/A\t";
        else
            out << r.GetMin() << "\t";
        if (r.GetMax() == FLT_MAX)
            out << "N/A\t";
        else
            out << r.GetMax() << "\t";

        out << endl;
    }
    out << endl
        << "Note: results marked with (*) had missing values such as" << endl
        << "might occur with a mixture of architectural capabilities." << endl;
}

// ****************************************************************************
//  Method:  ResultDatabase::ClearAllResults
//
//  Purpose:
//    Clears all existing results from the ResultDatabase; used for multiple passes
//    of the same test or multiple tests.
//
//  Arguments:
//
//  Programmer:  Jeffrey Young
//  Creation:    September 10th, 2014
//
//  Modifications:
//
//
// ****************************************************************************
void ResultDatabase::ClearAllResults() { results.clear(); }

// ****************************************************************************
//  Method:  ResultDatabase::DumpCsv
//
//  Purpose:
//    Writes either detailed or summary results (min/max/stddev/med/mean), but not
//    every individual trial.
//
//  Arguments:
//    out        file to print CSV results
//
//  Programmer:  Jeffrey Young
//  Creation:    August 28th, 2014
//
//  Modifications:
//
// ****************************************************************************
void ResultDatabase::DumpCsv(string fileName) {
    bool emptyFile;
    vector<Result> sorted(results);

    stable_sort(sorted.begin(), sorted.end());

    // Check to see if the file is empty - if so, add the headers
    emptyFile = this->IsFileEmpty(fileName);

    // Open file and append by default
    ofstream out;
    out.open(fileName.c_str(), std::ofstream::out | std::ofstream::app);

    // Add headers only for empty files
    if (emptyFile) {
        // TODO: in big parallel runs, the "trials" are the procs
        // and we really don't want to print them all out....
        out << "test, "
            << "atts, "
            << "units, "
            << "median, "
            << "mean, "
            << "stddev, "
            << "min, "
            << "max, ";
        out << endl;
    }

    for (int i = 0; i < sorted.size(); i++) {
        Result& r = sorted[i];
        out << r.test << ", ";
        out << r.atts << ", ";
        out << r.unit << ", ";
        if (r.GetMedian() == FLT_MAX)
            out << "N/A, ";
        else
            out << r.GetMedian() << ", ";
        if (r.GetMean() == FLT_MAX)
            out << "N/A, ";
        else
            out << r.GetMean() << ", ";
        if (r.GetStdDev() == FLT_MAX)
            out << "N/A, ";
        else
            out << r.GetStdDev() << ", ";
        if (r.GetMin() == FLT_MAX)
            out << "N/A, ";
        else
            out << r.GetMin() << ", ";
        if (r.GetMax() == FLT_MAX)
            out << "N/A, ";
        else
            out << r.GetMax() << ", ";

        out << endl;
    }
    out << endl;

    out.close();
}

// ****************************************************************************
//  Method:  ResultDatabase::IsFileEmpty
//
//  Purpose:
//    Returns whether a file is empty - used as a helper for CSV printing
//
//  Arguments:
//    file  The input file to check for emptiness
//
//  Programmer:  Jeffrey Young
//  Creation:    August 28th, 2014
//
//  Modifications:
//
// ****************************************************************************

bool ResultDatabase::IsFileEmpty(string fileName) {
    bool fileEmpty;

    ifstream file(fileName.c_str());

    // If the file doesn't exist it is by definition empty
    if (!file.good()) {
        return true;
    } else {
        fileEmpty = (bool)(file.peek() == ifstream::traits_type::eof());
        file.close();

        return fileEmpty;
    }

    // Otherwise, return false
    return false;
}


// ****************************************************************************
//  Method:  ResultDatabase::GetResultsForTest
//
//  Purpose:
//    Returns a vector of results for just one test name.
//
//  Arguments:
//    test       the name of the test results to search for
//
//  Programmer:  Jeremy Meredith
//  Creation:    December  3, 2010
//
//  Modifications:
//
// ****************************************************************************
vector<ResultDatabase::Result> ResultDatabase::GetResultsForTest(const string& test) {
    // get only the given test results
    vector<Result> retval;
    for (int i = 0; i < results.size(); i++) {
        Result& r = results[i];
        if (r.test == test) retval.push_back(r);
    }
    return retval;
}

// ****************************************************************************
//  Method:  ResultDatabase::GetResults
//
//  Purpose:
//    Returns all the results.
//
//  Arguments:
//
//  Programmer:  Jeremy Meredith
//  Creation:    December  3, 2010
//
//  Modifications:
//
// ****************************************************************************
const vector<ResultDatabase::Result>& ResultDatabase::GetResults() const { return results; }
