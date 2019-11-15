import os
import sys
import commands
import re
import xlsxwriter
import gc
import collections

ROCM_VERSION = "rocm" + os.popen('dkms status').readlines()[0].split(",")[1]
CUDA_VERSION = os.popen('cat /usr/local/cuda/version.txt').readlines()[0]

def get_data(filesource):

    #regex pattern for each section start
    start_global_pattern          = r"\[HIPIFY\] info: file \'GLOBAL\' statistics:"
    start_convert_type_pattern    = r"\[HIPIFY\] info: CONVERTED refs by type:"
    start_convert_API_pattern     = r"\[HIPIFY\] info: CONVERTED refs by API:"
    start_convert_name_pattern    = r"\[HIPIFY\] info: CONVERTED refs by names:"
    start_unconvert_type_pattern  = r"\[HIPIFY\] info: UNCONVERTED refs by type:"
    start_unconvert_API_pattern   = r"\[HIPIFY\] info: UNCONVERTED refs by API:"
    start_unconvert_name_pattern  = r"\[HIPIFY\] info: UNCONVERTED refs by names:"
    start_total_stats_pattern     = r"\[HIPIFY\] info: TOTAL statistics:"
    sectionPatterns = [start_global_pattern,
                       start_convert_type_pattern,
                       start_convert_API_pattern,
                       start_convert_name_pattern,
                       start_unconvert_type_pattern,
                       start_unconvert_API_pattern,
                       start_unconvert_name_pattern,
                       start_total_stats_pattern]


    #regex pattern for each line that contains data
    data_pattern = r"(.*): ([0-9\.]*)"

    #flag to each section 
    isGlobalStatsData   = False
    isConvertTypeData   = False
    isConvertAPIData    = False
    isConvertNameData   = False
    isUnconvertTypeData = False
    isUnconvertAPIData  = False
    isUnconvertNameData = False
    isTotalStatsData    = False
    sectionFlags = [isGlobalStatsData,
                    isConvertTypeData,
                    isConvertAPIData, 
                    isConvertNameData,
                    isUnconvertTypeData,
                    isUnconvertAPIData,
                    isUnconvertNameData,
                    isTotalStatsData]

    # execute test
    filep = open(filesource, "r")
    lines = filep.readlines()

    #dictionary for each section
    global_stats_database   = {}
    convert_type_database   = {}
    convert_API_database    = {}
    convert_name_database   = {}
    unconvert_type_database = {}
    unconvert_API_database  = {}
    unconvert_name_database = {}
    total_stats_database    = {}
    sectionDict = [global_stats_database,
                   convert_type_database,
                   convert_API_database,
                   convert_name_database,
                   unconvert_type_database,
                   unconvert_API_database, 
                   unconvert_name_database,
                   total_stats_database]

    #section counter
    sectionIndex = 0
   
    #number of sections
    numSections = 8

    for line in lines:
        if sectionFlags[sectionIndex] == False and sectionIndex == 0:
            result = re.match(sectionPatterns[sectionIndex], line)
            if result:
                sectionFlags[sectionIndex] = True
                print "Started Parsing " + filesource
        elif sectionFlags[sectionIndex] == True: 
            matchIndex = 1 
            while matchIndex < numSections:
                result = re.match(sectionPatterns[matchIndex], line)
                if result:
                    break
                matchIndex = matchIndex + 1
            if result:
                sectionFlags[matchIndex]   = True
                sectionFlags[sectionIndex] = False
                sectionIndex = matchIndex
            else:
                result = re.match(data_pattern, line)
                if result and result.group(2) != '':
                    try: 
                        sectionDict[sectionIndex][result.group(1).lstrip(" ")] = int(result.group(2))
                    except ValueError:
                        sectionDict[sectionIndex][result.group(1).lstrip(" ")] = float(result.group(2))
                    #print result.group(1).lstrip(" ") + ": " + result.group(2)

    filep.close() 
    return sectionDict 

def summingDict(listOfDict):
    finalSum = []
    
    numOfDict = len(listOfDict) 
    numOfSections = len(listOfDict[0])

    for i in range(0, numOfSections):
        finalSum.append({})

    dictIndex = 0 
    sectionIndex = 0

    for eachDict in listOfDict:
        for eachSect in eachDict:
            for node in eachSect:
                if node in finalSum[sectionIndex]:
                    finalSum[sectionIndex][node] = finalSum[sectionIndex][node] + eachSect[node]
                else:
                    finalSum[sectionIndex][node] = eachSect[node]
            sectionIndex += 1
        sectionIndex = 0
        dictIndex += 1
    return finalSum

def outputReport(workbook, allDict, pageName, sectionHeaders):

    title_prop = workbook.add_format({'bold':True, 'align':'center', 'fg_color':'#bcdbff', 'border':5})
    cell_prop = workbook.add_format({'border':5})
    highlight_prop = workbook.add_format({'bold':True,'fg_color':'#ff0000', 'border':5})

    width_node = 30
    width_data = 15
    column = 0
    column_start = 1
    row = row_start = 3 

    numWorksheets = 8
    worksheetIndex = 0
    currentWorksheet = workbook.add_worksheet(pageName)
    while worksheetIndex < 8:
        currentWorksheet.write(row, column, sectionHeaders[worksheetIndex][0], title_prop)
        currentWorksheet.write(row, column + 1, sectionHeaders[worksheetIndex][1], title_prop)
        row += 1

        if worksheetIndex == 0:
            dataDict = allDict[worksheetIndex]
            for node in dataDict:
                if node == "UNCONVERTED refs count":
                    currentWorksheet.write(row, column, node, highlight_prop)
                    currentWorksheet.write(row, column + 1, dataDict[node], highlight_prop)
                else:
                    currentWorksheet.write(row, column, node, cell_prop)
                    currentWorksheet.write(row, column + 1, dataDict[node], cell_prop)
                row += 1
        else:
            dataDict = sorted(allDict[worksheetIndex].items(), key=lambda d: d[1], reverse=True)
            for node in dataDict:
                if node == "UNCONVERTED refs count":
                    currentWorksheet.write(row, column, node[0], highlight_prop)
                    currentWorksheet.write(row, column + 1, node[1], highlight_prop)
                else:
                    currentWorksheet.write(row, column, node[0], cell_prop)
                    currentWorksheet.write(row, column + 1, node[1], cell_prop)
                row += 1
        worksheetIndex += 1
        currentWorksheet.set_column(column, column, width_node)
        currentWorksheet.set_column(column + 1, column + 1, width_data)
        currentWorksheet.set_column(column + 2, column + 2, width_data)
        column = column_start + 3 * worksheetIndex
        row = row_start
    return


def writeEachLog(targetDir, destFile, dictAllLogs):

    #Headers for each section
    globalStatsHeader   = ["Global Statistic Summary", "Value"]
    convertTypeHeader   = ["Converted CUDA References Type", "Frequencies"]
    convertAPIHeader    = ["Converted CUDA References API", "Frequencies"]
    convertNameHeader   = ["Converted CUDA References Name", "Frequencies"]
    unconvertTypeHeader = ["Unconvert CUDA References Type", "Frequencies"]
    unconvertAPIHeader  = ["Unconvert CUDA References API", "Frequencies"]
    unconvertNameHeader = ["Unconvert CUDA References Name", "Frequencies"]
    totalStatsHeader    = ["Total Statistic Summary", "Numbers"]
    sectionHeaders = [globalStatsHeader,
                      convertTypeHeader,
                      convertAPIHeader,
                      convertNameHeader,
                      unconvertTypeHeader,
                      unconvertAPIHeader,
                      unconvertNameHeader,
                      totalStatsHeader]

    excelbook = xlsxwriter.Workbook(destFile)

    fileNames = retrieveEachFileName(targetDir)
    
    overallDict = summingDict(dictAllLogs)

    outputReport(excelbook, overallDict, "Overall Summary", sectionHeaders)

    numDict = len(dictAllLogs)
    pageIndex = 0
    while pageIndex < numDict:
        outputReport(excelbook, dictAllLogs[pageIndex], fileNames[pageIndex], sectionHeaders)
        pageIndex += 1

    allUnconvertRefDict = retrieveAllUnconvertRef(fileNames, dictAllLogs)
    allConvertRefDict = retrieveAllConvertRef(fileNames, dictAllLogs)
    addAllUnconvertRefTable(excelbook, allUnconvertRefDict, allConvertRefDict)

    excelbook.close()
    print "Generating Report to " + destFile 
    return

def addAllUnconvertRefTable(workbook, allUnconvertRef, allConvertRef):
    title_prop = workbook.add_format({'bold':True, 'align':'center', 'fg_color':'#bcdbff', 'border':5})
    cell_prop = workbook.add_format({'border':5})
    highlight_prop = workbook.add_format({'bold':True,'fg_color':'#ff0000', 'border':5})

    width_node = 30
    width_data = 15
    column = column_start = 0
    row = row_start = 17 

    currentWorksheet = workbook.get_worksheet_by_name("Overall Summary")
    currentWorksheet.write(0, 0, ROCM_VERSION, title_prop)
    currentWorksheet.write(1, 0, CUDA_VERSION, title_prop)
    
    currentWorksheet.write(row, column, "CUDA refs by APP", title_prop)
    currentWorksheet.write(row, column + 1, "UNCONVERTED", title_prop)
    currentWorksheet.write(row, column + 2, "CONVERTED", title_prop)
    row += 1
    dataList = sorted(allUnconvertRef.items(), key=lambda d: d[1], reverse=True)
    for node in dataList:
        if node[1] != 0:
            currentWorksheet.write(row, column, node[0], highlight_prop)
            currentWorksheet.write(row, column + 1, node[1], highlight_prop)
            currentWorksheet.write(row, column + 2, allConvertRef[node[0]], highlight_prop)
        else:
            currentWorksheet.write(row, column, node[0], cell_prop)
            currentWorksheet.write(row, column + 1, node[1], cell_prop)
            currentWorksheet.write(row, column + 2, allConvertRef[node[0]], cell_prop)
        row += 1
    currentWorksheet.set_column(column, column, width_node)
    currentWorksheet.set_column(column + 1, column + 1, width_data)
    return

def retrieveAllConvertRef(eachFileName, allDataDict):
    allConvertRef = {}
    index = 0
    numDict = len(allDataDict)
    for eachDict in allDataDict:
        allConvertRef[eachFileName[index]] = eachDict[0]["CONVERTED refs count"]
        index += 1
    return allConvertRef

def retrieveAllUnconvertRef(eachFileName, allDataDict):
    allUnconvertRef = {}
    index = 0
    numDict = len(allDataDict)
    for eachDict in allDataDict:
        allUnconvertRef[eachFileName[index]] = eachDict[0]["UNCONVERTED refs count"]
        index += 1
    return allUnconvertRef

def retrieveEachFileName(targetDir):
    allLogs = [log for log
               in os.listdir(targetDir)
               if os.path.isfile(os.path.join(targetDir, log))]
    return allLogs

def processEachLog(targetDir):
    allLogs = [os.path.join(targetDir, log) for log
               in os.listdir(targetDir)
               if os.path.isfile(os.path.join(targetDir, log))]
    allLogs = sorted(allLogs, key=lambda s: s.lower())
    print allLogs
    dictAllLogs = []
    for eachLog in allLogs:
        dictAllLogs.append(get_data(eachLog))
    return dictAllLogs


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "The format for issuing parameters should be as the following:"
        print "\n"
        print "    python hipreport.py [HIPIFY_CLANG_LOG] [OUTPUT_XLSX_FILE]" 
        print "\n"
        print "Where each of the parameter's purpose are listed down below:"
        print "\n"
        print "[HIPIFY_CLANG_LOG]   -  The path to the directory that contains all the hipify-clang/hipexamined logs for processing."
        print "[OUTPUT_XLSX_FILE]   -  The path to the destination excel report file to be generated."
        print "\n"
        print "Example:"
        print "\n"
        print "    python GenHipifyXlsx.py /home/username/log/ report.xlsx"
        print "\n"
    elif os.path.isdir(sys.argv[1]) != True:
        print "\n"
        print "The first argument [HIPIFY_CLANG_LOG] is not a valid directory path"
        print "\n"
    elif sys.argv[2].endswith(".xlsx") != True:
        print "\n"
        print "The second argument [OUTPUT_XLSX_FILE] needs to containi .xlsx file extention"
        print "\n"
    else :
        listDataBase = processEachLog(sys.argv[1])
        writeEachLog(sys.argv[1], sys.argv[2], listDataBase)


