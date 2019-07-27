import statistics


# number of posts
def extract_number_of_posts(df):
    return statistics.posts_per_user(df)


# number of offensive posts
def extract_number_of_offensive_posts(df):
    return statistics.posts_per_user_and_class(df, 3)


# number of non offensive posts
def extract_number_of_non_offensive_posts(df):
    return statistics.posts_per_user_and_class(df, 1)


def extract_features(df):
    user_posts = extract_number_of_posts(df)
    user_offensive_posts = extract_number_of_offensive_posts(df)
    user_non_offensive_posts = extract_number_of_non_offensive_posts(df)
    user_posts_all = user_offensive_posts.merge(user_non_offensive_posts,
                                               on='writer',
                                               how='outer',
                                               suffixes=['_offensive', '_not_offensive']).fillna(0)
    return user_posts_all.merge(user_posts, on='writer')
