import seaborn as sns
from adjustText import adjust_text

def jbarplot(ax, df, chtitle, xlab, ylab, **kwargs):
    ax=sns.barplot(data=df, **kwargs)
    ax.legend()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(chtitle, fontsize=14)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_xlabel(xlab, fontsize=10)
    ax.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    return ax

def jbarplot_try(ax, df, chtitle, xlab, ylab, **kwargs):
    ax = (
        sns.barplot(data=df, **kwargs)
        .legend()
        .set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        .set_title(chtitle, fontsize=14)
        .set_ylabel(ylab, fontsize=10)
        .set_xlabel(xlab, fontsize=10)
        .grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    )
    return ax

def jlabelstripplot(ax, df, xcho, varcho, chtitle, xlab, ylab, **kwargs):
    ax = sns.stripplot(data=df, x=xcho, y=varcho, **kwargs)
    TEXTS = []
    for i, fname in enumerate(df.index):
        x=df[xcho][i]
        y=df[varcho][i]
        TEXTS.append(ax.text(x, y, fname, ha='right', va='center'))
    ax.set_title(chtitle, fontsize=14)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    atout = adjust_text(TEXTS, force_text=(1.2, 1.2), expand_points=(1.2, 1.2), arrowprops=dict(arrowstyle='-', color='grey'))
    return ax

def jbarplotnoleg(ax, df, chtitle, xlab, ylab, **kwargs):
    ax=sns.barplot(data=df, **kwargs)
    #ax.legend()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(chtitle, fontsize=14)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_xlabel(xlab, fontsize=10)
    ax.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    return ax

def jbarplotnoleg_try(ax, df, chtitle, xlab, ylab, **kwargs):
    ax= (
        sns.barplot(data=df, **kwargs)
        #.legend()
        .set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        .set_title(chtitle, fontsize=14)
        .set_ylabel(ylab, fontsize=10)
        .set_xlabel(xlab, fontsize=10)
        .grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    )
    return ax