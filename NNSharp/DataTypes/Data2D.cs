using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.DataTypes
{

    public class Data2D : IData
    {

        public Data2D(int height, int width, int channels, int batchSize)
        {
            tensor = new double[height, width, channels, batchSize];
            D = new Dimension(height, width, channels, batchSize);
            this.paddingValue = 0;
        }

        public Data2D(int height, int width, int channels, int batchSize, int paddingValue)
        {
            tensor = new double[height, width, channels, batchSize];
            D = new Dimension(height, width, channels, batchSize);
            this.paddingValue = paddingValue;
        }

        public double this[int h, int w, int c, int b]
        {
            get
            {
                double val = 0.0;

                if (h < 0 || h >= D.h)
                    val = paddingValue;
                else if (w < 0 || w >= D.w)
                    val = paddingValue;
                else if (c < 0 || c >= D.c)
                    val = paddingValue;
                else if (b < 0 || b >= D.b)
                    val = paddingValue;
                else
                    val = tensor[h, w, c, b];

                return val;
            }

            set
            {
                tensor[h, w, c, b] = value;
            }
        }

        public void ApplyToAll(Operation operation)
        {
            for (int b = 0; b < D.b; ++b)
            {
                for (int c = 0; c < D.c; ++c)
                {
                    for (int w = 0; w < D.w; ++w)
                    {
                        for (int h = 0; h < D.h; ++h)
                        {
                            tensor[h, w, c, b] = operation(tensor[h, w, c, b]);
                        }
                    }
                }
            }
        }

        public void ToZeros()
        {
            this.ApplyToAll(p => { return 0.0; });
        }

        public Dimension GetDimension()
        {
            return D;
        }

        private double[,,,] tensor;

        public struct Dimension
        {
            public Dimension(int h, int w, int c, int b)
            {
                this.h = h; this.w = w;
                this.c = c; this.b = b;
            }
            public int h; // height
            public int w;
            public int c;
            public int b;
        } private Dimension D;

        private int paddingValue;

    }
}
