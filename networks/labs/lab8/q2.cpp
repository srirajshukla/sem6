// Program to write and read two messages through the pipe using the parent and the child processes.
#include<iostream>
#include<unistd.h>
#include <array>

using std::cout;

int main() {
   std::array<int,2> pipe_file_descpritor;
   int returnstatus;
   int pid;

   char writemessages[2][20]={"Message", "To the World"};
   char readmessage[20];
   
   returnstatus = pipe(pipe_file_descpritor.data());
   if (returnstatus == -1) {
      cout<<"Unable to create pipe\n";
      return 1;
   }
   pid = fork();
   
   // Child process
   if (pid == 0) {
      read(pipe_file_descpritor[0], readmessage, sizeof(readmessage));
      cout << "Child Process - Reading from pipe – Message 1 is " << readmessage << "\n";
      read(pipe_file_descpritor[0], readmessage, sizeof(readmessage));
      cout << "Child Process - Reading from pipe – Message 2 is " << readmessage << "\n";
   } else { //Parent process
      cout << "Parent Process - Writing to pipe – Message 1 is " << writemessages[0] << "\n";
      write(pipe_file_descpritor[1], writemessages[0], sizeof(writemessages[0]));
      cout << "Parent Process - Writing to pipe – Message 2 is " << writemessages[1] << "\n";
      write(pipe_file_descpritor[1], writemessages[1], sizeof(writemessages[1]));
   }
   return 0;
}