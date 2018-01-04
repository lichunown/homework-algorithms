package question1;

import java.util.*;
import java.math.*;

public class Main {

	public static void main(String[] args) {
		int N = 10;
		int runTimes = 1;				
		int[] runForOpenNums = PercolationStats(N, runTimes);
		System.out.printf("[PercolationStats]  N=%d   runTimes=%d\n",N,runTimes);
		System.out.printf("\t mean:   %f\n",mean(runForOpenNums,N,runTimes));
		System.out.printf("\t stddev: %f\n",stddev(runForOpenNums,N,runTimes));
	}

	private static int runOneTime(int N) {
		Percolation temp = new Percolation(N);
		Random rand = new Random();
		while(!temp.isConnected()) {
			temp.open(rand.nextInt(N)+1, rand.nextInt(N)+1);// range(1,N+1)
		}
		temp.print();
		return temp.openNum();
	}
	public static int[] PercolationStats(int N, int runTimes) {
		int[] runForOpenNums = new int[runTimes];
		for(int i = 0;i<runTimes;i++) {
			runForOpenNums[i] = runOneTime(N);
		}
		return runForOpenNums;
	}
	public static int sum(int[] nums, int runTimes) {
		int sumn = 0;
		for(int i=0;i<runTimes;i++) {
			sumn += nums[i];
		}
		return sumn;
	}
	public static double mean(int[] runForOpenNums, int N, int runTimes) {
		return (double)sum(runForOpenNums,runTimes)/(double)(runTimes*N*N);
	}
	public static double stddev(int[] runForOpenNums, int N, int runTimes) {
		double mean = mean(runForOpenNums, N, runTimes);
		double temp = 0;
		for(int i=0; i<runTimes;i++) {
			temp += Math.abs(mean - runForOpenNums[i]/(double)(N*N));
		}
		return Math.sqrt(temp)/runTimes;
	}
	
}
