package question2;

import java.util.Random;

public class Main {
	public static int[] RandomInts(int num) {
		Random rand = new Random();
		int[] data = new int[num];
		for(int i = 0;i < num; i++) {
			data[i] = rand.nextInt(num);
		}
		return data;
	}
	public static void main(String[] args) {
		Comparable[] data = RandomInts(10);
		Sort sort  = new Sort();
		sort.show(data);
		System.out.println();
		sort.Sort_Insert(data);
		sort.show(data);
	}
}
