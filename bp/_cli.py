import click
from bp import parse_jplace, insert_fully_resolved

try:
    from bp.GPL import insert_multifurcating
except:
    insert_multifurcating = None

@click.group()
def cli():
    pass


@cli.command()
@click.option('--placements', type=click.Path(exists=True),
              required=True, help='jplace formatted data')
@click.option('--output', type=click.Path(exists=False),
              required=True, help='Where to write the resulting newick')
@click.option('--method',
              type=click.Choice(['fully-resolved', 'multifurcating']),
              required=True, help='Whether to fully resolve or multifurcate')
def placement(placements, output, method):
    if method == 'fully-resolved':
        f = insert_fully_resolved
    elif method == 'multifurcating':
        if insert_multifurcating is None:
            raise ValueError("Please install with 'pip install iow-gpl'")
        f = insert_multifurcating
    else:
        raise ValueError("Unknown method: %s" % method)

    placement_df, tree = parse_jplace(open(placements).read())
    sktree = f(placement_df, tree)
    sktree.write(output)


if __name__ == '__main__':
    cli()
