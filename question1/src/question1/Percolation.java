package question1;

public class Percolation {
	private int N;
	private int[] uniondata;
	private boolean[] opendata;//true -> open ; false -> not open
	public Percolation(int N) {
		this.N = N;	
		this.uniondata = new int[N*N+2];
		this.opendata = new boolean[N*N+2];
		for(int i=0; i<N*N+2; i+=1) {
			this.uniondata[i] = i;
			this.opendata[i] = false;
		}
		this.opendata[getFirstID()] = true;
		this.opendata[getFinalID()] = true;
	}
	
	
	private int find(int p) {
		while(p != uniondata[p])p = uniondata[p];
		return p;
	}

	private void union(int p, int q) {
		int pRoot = find(p);
		int qRoot = find(q);
		if(pRoot == qRoot) return;
		uniondata[pRoot] = qRoot;
	}
	
	public void open(int i,int j) {
		if(isOpen(i,j) == false) {
			opendata[getID(i,j)] = true;
			if(i - 1 == 0) {
				union(getID(i,j), getFirstID());
			}
			if(i == N){
				union(getID(i,j), getFinalID());
			}
			if(i + 1 <= N)
				if(isOpen(i+1,j))
					union(getID(i,j), getID(i+1,j));			
			if(i - 1 >= 1)
				if(isOpen(i-1,j))
					union(getID(i,j), getID(i-1,j));
			
			if(j + 1 <= N)
				if(isOpen(i,j+1))
					union(getID(i,j), getID(i,j+1));	
			if(j - 1 >= 1)
				if(isOpen(i,j-1))
					union(getID(i,j), getID(i,j-1));			
		}
	}
	
	public boolean isConnected() {
		return find(getFirstID()) == find(getFinalID());
	}
	public boolean isConnected(int i,int j) {
		return find(getFirstID()) == find(getID(i,j));
	}
	public boolean isOpen(int i,int j) {
//		System.out.printf("i:%d j:%d   id:%d\n",i,j,getID(i,j));
		return opendata[getID(i,j)];
	}
	public void print() {
		for(int i = 1; i <= N; i++) {
			for(int j = 1; j <= N; j++) {
				if(isConnected(i,j)) {
					System.out.printf("* ");
				}else if(isOpen(i,j)) {
					System.out.printf("o ");
				}else{
					System.out.printf("# ");
				}	
			}
			System.out.printf("\n");
		}
	}
	private int getFirstID() {
		return 0;
	}
	private int getFinalID() {
		return N*N+2-1;
	}	
	public int openNum() {
		int opencount = 0;
		for(int i=1;i<=N;i++) {
			for(int j=1;j<=N;j++) {
				if(isOpen(i,j)) {
					opencount += 1;		
				}			
			}
		}
		return 	opencount;	
	}
	public double openPercent() {
		int opencount = 0;
		for(int i=1;i<=N;i++) {
			for(int j=1;j<=N;j++) {
				if(isOpen(i,j)) {
					opencount += 1;		
//					System.out.println("??");
				}			
			}
		}
		return (double)(opencount)/(double)(N*N);
	}
	private int getID(int i,int j) {
		return (i-1)*N+j;
	}	
//	private int getUnion(int id) {
//		return find(id);
//	}		
}
