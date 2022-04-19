#include <bits/stdc++.h>
#define ll long long
#define srt(s) sort(s.begin(), s.end())
#define sf(s) s.begin(), s.end()
#define iofile true
using namespace std;

int main(){
    #ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    #endif

    #if !defined(iofile) || defined(ONLINE_JUDGE)
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    #endif

    int t;
    cin>>t;
    while(t--){
        int n;
        cin>>n;

        vector<int> x(n), y(n);

        for(int i=0; i<(4*n-1); i++){
            cin>>x[i]>>y[i];
        }
        srt(x);
        srt(y);

        int xc = x[0], yc = y[0];
        for(int i=0; i<n; i++){
            int j=i+1;
            for(; j<n; j++){
                if (x[j]==x[j-1]){
                    j++;
                } else{
                    break;
                }
            }

            int cons = j-i;
            if (cons%2==1){
                xc = x[i];
                break;
            }
            i = j-1;
        }

        for(int i=0; i<n; i++){
            int j=i+1;
            for(; j<n; j++){
                if (y[j]==y[j-1]){
                    j++;
                } else{
                    break;
                }
            }

            int cons = j-i;
            if (cons%2==1){
                yc = y[i];
                break;
            }
            i = j-1;
        }

        cout << xc << " " << yc << endl;
    }
    return 0;
}