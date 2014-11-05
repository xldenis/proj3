# series of layers
#   each neuron connected to next layer
#     each connection has weight
#     value of neurons given by sigmoid(sum(w*(prev layer output)+b))

def sigmoid(x)
  1 / (1 + Math.exp(-x))
end

class FFNN 

  def initialize(layers)
    @weights = (layers.size - 1).times.map do |i|
      layers[i+1].times.map do |neuron|
        layers[i].times.map {rand}
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
    outputs = [example]
    outputs += @weights.map do |mat|
      vec = mat.map do |row|
        sigmoid(vec.zip(row).map {|a,b| a*b}.flatten.reduce(:+))
      end
    end
    delta = []
    delta[@weights.size-1] = label.zip(outputs.last).map {|l,o| o*(1-o)*(l - o)}
    (@weights.size() -2).downto(0).each do |l|
      vec = outputs[l+1]
      layer_update = @weights[l+1].transpose.map {|r| r.zip(delta[l+1]).map{|a,b| a*b}.reduce(:+)}
      delta[l] = vec.zip(layer_update).map {|o, u| o*(1-o)*u}
    end
    # puts delta.to_s
    #update
    @weights = @weights.each_with_index.map do |mat,i|
      mat = mat.each_with_index.map do |row,j|
        row.each_with_index.map do |n,k|
          n +0.8 * outputs[i][k] * delta[i][j]
        end
      end
    end
    # puts @weights.last.to_s
  end 
end

ann = FFNN.new([8,3,8])
puts ann.evaluate([0,0,0,0,1,0,0,0]).to_s
400.times do |i|
end
