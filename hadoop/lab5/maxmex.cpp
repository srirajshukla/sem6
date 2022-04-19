// https://www.codechef.com/LP2TO302/status/MEX

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
        ll n, k;
        cin>>n>>k;
        vector<ll> vals(n);
        for(auto &val:vals)cin>>val;

        srt(vals);
    
        for(int i=0; i<vals.size(); i++){
            if (vals[i]!=i){
                if (k>0){
                    vals.insert(vals.begin()+i, i);
                    k--;
                } else{
                    break;
                }
            }
        }

        ll ans=n;
        for(int i=0; i<vals.size(); i++){
            if (vals[i]!=i){
                ans = i;
                break;
            }
        }

        if (k>0)
            ans += k;

        cout<<ans<<endl;
    }
    return 0;
}