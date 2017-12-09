package question2;

import java.lang.Comparable;

public class Sort {
	public Sort() {
		
	}
	public static void Sort_Insert(Comparable[] a) {
		int N = a.length;
		for(int i = 0; i < N; i++) {
			int min = i;
			for(int j = i + 1; j < N; j++) {
				if(less(a[j],a[min])) min = i;
			}
			exch(a, i ,min);
		}
	}
	
	private static void exch(Comparable[] a, int i, int j) {
		Comparable t = a[i];
		a[i] = a[j];
		a[j] = t;
	}
	public static boolean less(Comparable a, Comparable b) {
		return a.compareTo(b) < 0;
	}
	
	public static void show(Comparable[] a) {
		for(int i = 0; i < a.length; i++) {
			System.out.print(a[i]+" ");
		}
		System.out.print("\n");
	}
	public static boolean isSorted(Comparable[] a) {
		for(int i = 1; i< a.length; i++) {
			if(less(a[i],a[i-1])) return false;
		}
		return true;
	}
}
