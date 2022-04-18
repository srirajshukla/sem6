// Program to write and read two messages using pipe.
#include <iostream>
#include <unistd.h>
#include <array>

using std::cout;
int main()
{
    int pipe_file_descriptors[2];
    int returnstatus;
    char writemessages[2][20] = {"Message", "To Earth"};
    char readmessage[20];
    returnstatus = pipe(pipe_file_descriptors);

    if (returnstatus == -1){
        cout << "Unable to create pipe\n";
        return 1;
    }

    cout<<"Writing to pipe - Message 1 is "<<  writemessages[0] << "\n";
    write(pipe_file_descriptors[1], writemessages[0], sizeof(writemessages[0]));

    read(pipe_file_descriptors[0], readmessage, sizeof(readmessage));
    cout<<"Reading from pipe – Message 1 is " <<readmessage << "\n";

    cout<<"Writing to pipe - Message 2 is " << writemessages[1] <<"\n";
    write(pipe_file_descriptors[1], writemessages[1], sizeof(writemessages[1]));

    read(pipe_file_descriptors[0], readmessage, sizeof(readmessage));
    cout<<"Reading from pipe – Message 2 is "<< readmessage <<"\n";
}