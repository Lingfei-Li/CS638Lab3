package lingfei.CS638.Lab3.Data; /**
 * @Author: Yuting Liu
 * This is the dataset class that holds in the whole dataset
 * 
 */

import java.util.ArrayList;

public class Dataset {
	// the list of all instances
	private ArrayList<Instance> instances;

    public ArrayList<Instance> getImages() { return this.instances; }
	
	public Dataset() {
		this.instances = new ArrayList<Instance>();
	}
	
	// get the size of the dataset
	public int getSize() {
		return instances.size();
	}
	
	// add instance into the data set
	public void add(Instance inst) {
		instances.add(inst);
	}

	public void add(Dataset ds) {
		for(Instance inst : ds.instances) {
			instances.add(inst);
		}
	}
}
