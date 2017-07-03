/**
 *  @defgroup HipDb HCC-specific debug facilities
 *  @{
 */


/**
 * @brief * Print memory tracker information for this pointer.
 *
 * HIP maintains a table for all memory allocations performed by the application.
 * If targetAddress is 0, the entire table is printed to stderr.
 * If targetAddress is non-null, this routine will perform some forensic analysis
 * to find the pointer 
 */
void hipdbPrintMem(void *targetAddress);



// doxygen end HipDb
/**
 *   @}
 */
