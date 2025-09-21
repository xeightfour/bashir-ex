#include <iostream>

#define siz(v) (int)(v).size()
#define all(v) (v).begin(),(v).end()
#define bit(n, k) (((n) >> (k)) & 1)

using namespace std;

const int maxn = 1e5 + 10;
int n, q, out[maxn], in[maxn];

signed main() {
	ios::sync_with_stdio(false); cin.tie(nullptr);

	cin >> n >> q;
	while (q--) {
		int a, b;
		cin >> a >> b;
		out[a]++;
		in[b]++;
	}

	int res = -1;
	for (int i = 0; i < n; i++) {
		if (out[i] == n-1 and in[i] == 0) {
			res = i;
		}
	}

	cout << res << '\n';

	return 0;
}
