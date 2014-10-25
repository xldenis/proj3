# series of layers
#   each neuron connected to next layer
#     each connection has weight
#     value of neurons given by sigmoid(sum(w*(prev layer output)+b))

def sigmoid(x)
  1 / (1 + Math.exp(-x))
end

layers = [5,5,5,5,1]
weights = (layers.size - 1).times.map do |i|
  layers[i+1].times.map do |neuron|
    layers[i].times.map {1}
  end
end

def evaluate(example, weights)
  vec = example
  weights.each do |mat|
    vec = mat.map do |row|
      sigmoid(vec.zip(row).map {|a,b| a*b}.flatten.reduce(:+))
    end
  end
  vec
end

class FFNN 

  def initialize(layers)
    @weights = (layers.size - 1).times.map do |i|
      layers[i+1].times.map do |neuron|
        layers[i].times.map {1}
      end
    end
  end

  def evaluate(example)
    vec = example
    @weights.each do |mat|
      vec = mat.map do |row|
        sigmoid(vec.zip(row).map {|a,b| a*b}.flatten.reduce(:+))
      end
    end
    vec
  end

  def backprop(example, label)
    vec = example
    outputs = @weights.map do |mat|
      vec = mat.map do |row|
        sigmoid(vec.zip(row).map {|a,b| a*b}.flatten.reduce(:+))
      end
    end
    delta = []
    delta[@weights.size-1] = label.zip(outputs[0]).map {|l,o| o*(1-o)*(l - o)}
    (@weights.size() -2).downto(0).each do |l|
      vec = outputs[l]
      layer_update = @weights[l+1].transpose.map {|r| r.zip(delta[l+1]).map{|a,b| a*b}.reduce(:+)}
      delta[l] = vec.zip(layer_update).map {|o, u| o*(1-o)*u}
    end
    #update
    @weights = @weights.each_with_index.map do |mat,i|
      # puts i 
      mat = mat.map do |row|
        row.map {|n| n + delta[i].zip(outputs[i]).map {|a,b| a*b}.reduce(:+)}
      end
    end
  end 
end

ann = FFNN.new([5,5,5,1])

puts ann.evaluate([1,1,1,1,1]).to_s