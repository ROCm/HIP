#ifndef COMMON_H
#define COMMON_H

#include <errno.h>
#include <fstream>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <chrono>

#if defined(_WIN32)
#include <io.h>
#else
#include <error.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
class CaptureStream {
private:
  FILE* stream;
  int fdPipe[2];
  int fd;

  static constexpr size_t bufferSize = 25 * 1024 * 1024;

public:
  CaptureStream(FILE *original) {
    stream = original;

    if (pipe(fdPipe, bufferSize, O_TEXT) != 0) {
      fprintf(stderr, "pipe(3) failed with error %d\n", errno);
      assert(false);
    }

    if ((fd = dup(fileno(stream))) == -1) {
      fprintf(stderr, "dup(1) failed with error %d\n", errno);
      assert(false);
    }
  }

  ~CaptureStream() {
    close(fd);
    close(fdPipe[1]);
    close(fdPipe[0]);
  }

  void Begin() {
    fflush(stream);

    if (dup2(fdPipe[1], fileno(stream)) == -1) {
      fprintf(stderr, "dup2(2) failed with error %d\n", errno);
      assert(false);
    }

    setvbuf(stream, NULL, _IONBF, 0);
  }

  void End() {
    if (dup2(fd, fileno(stream)) == -1) {
      fprintf(stderr, "dup2(2) failed with error %d\n", errno);
      assert(false);
    }
  }

  std::string getData() {
    std::string data;
    data.resize(bufferSize);

    int numRead = read(fdPipe[0], const_cast<char*>(data.c_str()), bufferSize);
    data[numRead] = '\0';

    data.resize(strlen(data.c_str()));
    data.shrink_to_fit();

    return data;
  }
};
#else
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
  }

  void Begin() {
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

  void End() {
    fflush(nullptr);
    if (dup2(saved_fd, orig_fd) == -1) {
      error(0, errno, "Error");
      assert(false);
    }
    if (close(saved_fd) != 0) {
      error(0, errno, "Error");
      assert(false);
    }
  }

  std::string getData() {
    std::ifstream tmpFileStream(tempname);
    std::stringstream strStream;
    strStream << tmpFileStream.rdbuf();
    return strStream.str();
  }

  ~CaptureStream() {
    if (remove(tempname) != 0) {
      error(0, errno, "Error");
      assert(false);
    }
  }
};
#endif

#define DECLARE_DATA()                                                         \
  const char *msg_short = "Carpe diem.";                                       \
  const char *msg_long1 = "Lorem ipsum dolor sit amet, consectetur nullam. "   \
                          "In mollis imperdiet nibh nec ullamcorper.";         \
  const char *msg_long2 = "Curabitur nec metus sit amet augue vehicula "       \
                          "ultrices ut id leo. Lorem ipsum dolor sit amet, "   \
                          "consectetur adipiscing elit amet.";

#endif
