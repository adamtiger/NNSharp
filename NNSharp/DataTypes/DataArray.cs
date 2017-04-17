using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.DataTypes
{
    public class DataArray : IData, IEnumerable<double>
    {

        public DataArray(int length)
        {
            this.length = length;
            array = new double[length];
            ToZeros();
        }

        public double this[int idx]
        {
            get { return array[idx];}

            set { array[idx] = value; }
        }

        public void ApplyToAll(Operation operation)
        {
            for (int i = 0; i < length; ++i)
                array[i] = operation(array[i]);             
        }

        public void ToZeros()
        {
            for (int i = 0; i < length; ++i)
                array[i] = 0.0;
        }

        public IEnumerator<double> GetEnumerator()
        {
            var ret = ((IEnumerable<double>)array).GetEnumerator();
            return ret;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            var ret = (IEnumerator)array.GetEnumerator();
            return ret;
        }

        public int GetLength()
        {
            return length;
        }

        private double[] array;
        private int length;
    }
}
