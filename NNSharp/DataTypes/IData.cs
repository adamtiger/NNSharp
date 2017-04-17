using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.DataTypes
{
    public delegate double Operation(double current);

    public interface IData
    {
        void ApplyToAll(Operation operation);

        void ToZeros();
    }
}
