/* System call checks
   Copyright (C) 2025  Peter Behroozi

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
   If so, it should be in a file called "LICENSE".
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/errno.h>
#include <unistd.h>
#include <signal.h>
#include "check_syscalls.h"
#ifdef __linux__
#include <malloc.h>
#endif /* __linux__ */

//#define DEBUG_IO

char *unread = NULL;
int64_t unread_size = 0;
FILE *syscall_logfile = NULL;

#define SL ((syscall_logfile) ? syscall_logfile : stderr)

void reason_and_exit(void) {
  fprintf(SL, "[Error] Reason: %s\n", strerror(errno));
  exit(EXIT_FAILURE);
}

void system_error(char *errmsg) {
  if (errmsg) fprintf(SL, "[Error] %s\n", errmsg);
  reason_and_exit();
}

pid_t check_fork(void) {
  pid_t pid = fork();
  if (pid < 0)
    system_error("Fork failed.");
  return pid;
}

pid_t check_waitpid(pid_t pid) {
  int stat_loc;
  pid_t res;
  do {
    res = waitpid(pid, &stat_loc, 0);
  } while ((res < 0) && (errno == EINTR));

  if (res < 0) system_error("Waiting for child process failed.");
  return res;
}

FILE *check_fopen(char *filename, char *mode) {  
  FILE *res = fopen(filename, mode);
  if (res == NULL) {
    if (mode[0] == 'w')
      fprintf(SL, "[Error] Failed to open file %s for writing!\n", filename);
    else if (mode[0] == 'a')
      fprintf(SL, "[Error] Failed to open file %s for appending!\n", filename);
    else
      fprintf(SL, "[Error] Failed to open file %s for reading!\n", filename);
    reason_and_exit();
  }
#ifdef DEBUG_IO
  fprintf(SL, "[Note] Opened file %s with mode '%s' in fileno %d.\n", 
	  filename, mode, fileno(res));
#endif /* DEBUG_IO */
  return res;
}

FILE *check_fdopen(int fd, char *mode) {  
  FILE *res = fdopen(fd, mode);
  if (res == NULL) {
    if (mode[0] == 'w')
      fprintf(SL, "[Error] Failed to open fd %d for writing!\n", fd);
    else if (mode[0] == 'a')
      fprintf(SL, "[Error] Failed to open fd %d for appending!\n", fd);
    else
      fprintf(SL, "[Error] Failed to open fd %d for reading!\n", fd);
    reason_and_exit();
  }
#ifdef DEBUG_IO
  fprintf(SL, "[Note] Opened file descriptor %d with mode '%s'.\n",
	  fd, mode);
#endif /* DEBUG_IO */
  return res;
}


FILE *check_create_fopen(char *filename, char *mode) {
  FILE *res = fopen(filename, mode);
  if (res == NULL) {
    res = check_fopen(filename, "a");
    fclose(res);
    return check_fopen(filename, mode);    
  }
  #ifdef DEBUG_IO
  fprintf(SL, "[Note] Opened file %s with mode '%s' in fileno %d.\n", 
	  filename, mode, fileno(res));
#endif /* DEBUG_IO */
  return res;
}

FILE *check_popen(char *command, char *mode) {
  FILE *res = popen(command, mode);
  if (res == NULL) {
    fprintf(SL, "[Error] Failed to start command %s!\n", command);
    exit(EXIT_FAILURE);
  }
#ifdef DEBUG_IO
  fprintf(SL, "[Note] Opened command %s with mode '%s' in fileno %d.\n", 
	  command, mode, fileno(res));
#endif /* DEBUG_IO */
  return res;
}

FILE *check_rw_socket(char *command, pid_t *pid) {
  int sockets[2], status;
  pid_t wres;
  FILE *res;
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets)<0)
    system_error("Failed to create socket pair!");
  *pid = fork();
  if (*pid < 0) system_error("Failed to fork new process!");
  if (!*pid) {
    if (dup2(sockets[1], 0) < 0) system_error("Failed to reopen stdin!");
    if (dup2(sockets[1], 1) < 0) system_error("Failed to reopen stdout!");
    close(sockets[0]);
    if (execlp("sh", "sh", "-c", command, NULL) < 0)
      system_error("Failed to exec command!");
  }
  close(sockets[1]);
  res = fdopen(sockets[0], "r+");
  if (!res) system_error("Failed to convert socket to stream!");
  do {
    wres = waitpid(*pid, &status, WNOHANG);
  } while ((wres < 0) && (errno == EINTR));
  if (wres < 0) {
    fprintf(SL, "[Error] Failed to start child process: %s\n", command);
    exit(EXIT_FAILURE);
  }
#ifdef DEBUG_IO
  fprintf(SL, "[Note] Started command %s with mode 'r+' in fileno %d.\n", 
	  command, fileno(res));
#endif /* DEBUG_IO */
  return res;
}

void check_lseek(int fd, off_t offset, int whence) {
  int64_t res = lseek(fd, offset, whence);
  if (res<0) {
    fprintf(SL, "[Error] Lseek error in fileno %d: ", fd);
    fprintf(SL, "[Error] Reason: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
}

void rw_socket_close(FILE *res, pid_t pid) {
  fclose(res);
  kill(pid, 9);
  check_waitpid(pid);
}

void *check_realloc(void *ptr, size_t size, char *reason) {
  if (size > 0) {
    void *res = realloc(ptr, size);
    if (res == NULL) {
      fprintf(SL, "[Error] Failed to allocate memory (%s)!\n", reason);
      exit(EXIT_FAILURE);
    }
    return res;
  }
  if (ptr != NULL) free(ptr);
  return(NULL);
}


void *check_realloc_aligned(void *ptr, size_t prev_size, size_t size, char *reason) {
  if (size > 0) {
    void *res = check_realloc(ptr, size, reason);
    int64_t blocksize = 1<<11;
    if ((size > 100e6) && (prev_size > size) && (((int64_t)res) % blocksize)) {
      if (size % blocksize) size += blocksize - (size % blocksize);
      void *res2 = NULL;
      int rval = posix_memalign(&res2, blocksize, size);
      if (rval != 0) system_error("Failed to allocate aligned memory!\n");
      int64_t block = size / blocksize;
      for (; block >=0; block--) {
	int64_t min = block*blocksize;
	int64_t to_copy = prev_size - min;
	prev_size = min;
	if (!to_copy) continue;
	memcpy(res2+min, res+min, to_copy);
	res = check_realloc(res, min, reason);
      }
      res = res2;
    }
    return res;
  }
  if (ptr != NULL) free(ptr);
  return(NULL);
}

void _io_err(int rw, size_t size, size_t nitems, FILE *stream) {
  char *verb = (rw) ? "write" : "read";
  char *dir = (rw) ? "to" : "from";
  char *items = (nitems == 1) ? "item" : "items";

  fprintf(SL, "[Error] Failed to %s %"PRIu64" %s of size "
	  "%"PRIu64" bytes %s fileno %d!\n", 
	  verb, (uint64_t)nitems, items, (uint64_t)size, dir, fileno(stream));
  if (feof(stream))
    fprintf(SL, "[Error] Reason: end of file (offset %"PRIu64").\n",
	    (uint64_t)ftello(stream));
  else
    fprintf(SL, "[Error] Reason: %s\n", strerror(errno));
  exit(EXIT_FAILURE);
}

void check_fseeko(FILE *stream, off_t offset, int whence) {
  if (fseeko(stream, offset, whence) < 0) {
    fprintf(SL, "[Error] Seek error in fileno %d: ", fileno(stream));
    fprintf(SL, "[Error] Reason: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
}

//Works even for pipes
void check_fskip(FILE *stream, off_t offset, char *buffer, size_t buf_size) {
  int64_t n = 0;
  while (n<offset) {
    int64_t to_read = offset-n;
    if (buf_size < to_read) to_read = buf_size;
    n += check_fread(buffer, 1, to_read, stream);
  }
}

void check_limited_funread(void *ptr, size_t size, size_t nitems) {
  if (unread_size) {
    fprintf(SL, "[Error] Tried to unread twice in a row\n");
    exit(EXIT_FAILURE);
  }
  check_realloc_s(unread, size, nitems);
  unread_size = size*nitems;
  memcpy(unread, ptr, unread_size);  
}

size_t check_fread(void *ptr, size_t size, size_t nitems, FILE *stream) {
  size_t res = 1, nread = 0;
  if (unread_size) {
    if (unread_size != (size*nitems)) {
      fprintf(SL, "[Error] funread must be followed by identical fread!\n");
      exit(EXIT_FAILURE);
    }
    memcpy(ptr, unread, unread_size);
    check_realloc_s(unread, 0, 0);
    unread_size = 0;
    return nitems;
  }

  while (nread < nitems) {
    res = fread(ptr, size, nitems-nread, stream);
    if (res <= 0) _io_err(0, size, nitems, stream);
    nread += res;
    ptr = ((char *)ptr) + res*size;
  }
  return nread;
}

char *check_fgets(char *ptr, size_t size, FILE *stream) {
  char *res = fgets(ptr, size, stream);
  if (!res) _io_err(0, size, 1, stream);
  return res;
}

char *check_fgets_and_chomp(char *ptr, size_t size, FILE *stream) {
  char *res = check_fgets(ptr, size, stream);
  int64_t len = strlen(res);
  if (len>0 && res[len-1]=='\n') res[len-1] = 0;
  return res;
}


size_t check_fwrite(void *ptr, size_t size, size_t nitems, FILE *stream) {
  size_t res = 1, nwritten = 0;
  while (nwritten < nitems) {
    res = fwrite(ptr, size, nitems, stream);
    if (res <= 0) _io_err(1, size-1, nitems, stream);
    nwritten += res;
  }
  return nwritten;
}

void check_mkdir(char *dirname, mode_t mode) {
  if (mkdir(dirname, mode)==0) return;
  if (errno == EEXIST) return;
  fprintf(SL, "[Error] Failed to make directory %s!\n", dirname);
  system_error(NULL);
}

void check_fstat(int fd, struct stat *ts, char *filename) {
  if (fstat(fd, ts)!=0) {
    fprintf(SL, "[Error] Fstat failure on file %s!\n", filename);
    fprintf(SL, "[Error] Reason: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
}

void *check_mmap_file(char *filename, char mode, int64_t *length) {
  FILE *tf;
  int flags = MAP_SHARED, prot = PROT_READ;
  struct stat ts;
  if (mode == 'r' || mode=='c') tf = check_fopen(filename, "rb");
  else if (mode == 'w') {
    tf = check_fopen(filename, "r+b");
    prot |= PROT_WRITE;
  }
  else {
    fprintf(SL, "[Error] Invalid mode %c passed to check_mmap_file!\n", mode);
    exit(EXIT_FAILURE);
  }
  if (mode == 'c') {
    prot |= PROT_WRITE;
    flags = MAP_PRIVATE;
  }
  
  int fd = fileno(tf);
  check_fstat(fd, &ts, filename);

  void *res = NULL;
  if (ts.st_size > 0) {
    res = mmap(NULL, ts.st_size, prot, flags, fd, 0);
    if (res == MAP_FAILED) {
      fprintf(SL, "[Error] Mmap failure on file %s, mode %c!\n", filename, mode);
      fprintf(SL, "[Error] Reason: %s\n", strerror(errno));
      exit(EXIT_FAILURE);
    }
  }
  fclose(tf);
  if (length) *length = ts.st_size;
  return res;
}


void check_slurp_file(char *fn, void **data, int64_t *length) {
  FILE *input = check_fopen(fn, "r");
  int fd = fileno(input);
  struct stat ts = {0};
  check_fstat(fd, &ts, fn);
  *length = ts.st_size;
  if (ts.st_size < 1) return;
  check_realloc_s(*data, *length, 1);  
  if (read(fd, *data, *length) != (*length)) {
    fprintf(SL, "[Error] Failed to read %"PRId64" bytes from file %s!\n", *length, fn);
    reason_and_exit();
  }
}


#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif /* MAP_ANONYMOUS */

void *check_mmap_memory(int64_t length) {
  int flags = MAP_SHARED | MAP_ANONYMOUS, prot = PROT_READ | PROT_WRITE;
  void *res = mmap(NULL, length, prot, flags, -1, 0);
  if (res == MAP_FAILED) {
    fprintf(SL, "[Error] Mmap failure to allocate %"PRId64" bytes of memory!\n", length);
    fprintf(SL, "[Error] Reason: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
  return res;
}

void check_mtrim(void) {
#ifdef __linux__
  malloc_trim(0);
#endif /* __linux__ */
}

char *check_strdup(const char *str) {
  char *res = strdup(str);
  if (!res) 
    system_error("Memory allocation failure");
  return res;
}

void check_rename(const char *old, const char *new) {
  int64_t res = rename(old, new);
  if (res < 0) {
    fprintf(SL, "[Error] Rename of \"%s\" to \"%s\" failed.\n", old, new);
    system_error(NULL);
  }
}

void check_pipe(int filedes[2]) {
  int64_t res = pipe(filedes);
  if (res<0) system_error("Unable to create pipe!");
}

