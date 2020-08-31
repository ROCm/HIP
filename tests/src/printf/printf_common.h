#ifndef COMMON_H
#define COMMON_H

#include <errno.h>
#include <error.h>
#include <fstream>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <string>
#include <unistd.h>

struct CaptureStream {
  int saved_fd;
  int orig_fd;
  int temp_fd;

  char tempname[13] = "mytestXXXXXX";

  CaptureStream(FILE *original) {
    orig_fd = fileno(original);
    saved_fd = dup(orig_fd);

    if ((temp_fd = mkstemp(tempname)) == -1) {
      error(0, errno, "Error");
      assert(false);
    }

    fflush(nullptr);
    if (dup2(temp_fd, orig_fd) == -1) {
      error(0, errno, "Error");
      assert(false);
    }
    if (close(temp_fd) != 0) {
      error(0, errno, "Error");
      assert(false);
    }
  }

  void restoreStream() {
    if (saved_fd == -1)
      return;
    fflush(nullptr);
    if (dup2(saved_fd, orig_fd) == -1) {
      error(0, errno, "Error");
      assert(false);
    }
    if (close(saved_fd) != 0) {
      error(0, errno, "Error");
      assert(false);
    }
    saved_fd = -1;
  }

  std::ifstream getCapturedData() {
    restoreStream();
    std::ifstream temp(tempname);
    return temp;
  }

  ~CaptureStream() {
    restoreStream();
    if (remove(tempname) != 0) {
      error(0, errno, "Error");
      assert(false);
    }
  }
};

static std::string gulp(std::ifstream &input) {
  std::string retval;
  input.seekg(0, std::ios_base::end);
  retval.resize(input.tellg());
  input.seekg(0, std::ios_base::beg);
  input.read(&retval[0], retval.size());
  input.close();
  return retval;
}

#define DECLARE_DATA()                                                         \
  const char *msg_short = "Carpe diem.";                                       \
  const char *msg_long1 = "Lorem ipsum dolor sit amet, consectetur nullam. "   \
                          "In mollis imperdiet nibh nec ullamcorper.";         \
  const char *msg_long2 = "Curabitur nec metus sit amet augue vehicula "       \
                          "ultrices ut id leo. Lorem ipsum dolor sit amet, "   \
                          "consectetur adipiscing elit amet.";

#endif
